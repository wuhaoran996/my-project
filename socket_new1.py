# 多次之后再训练 添加经验回放池
# 单步 奖励

import socket
import threading
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math
import os
import json
import matplotlib.pyplot as plt  # 导入可视化库
import pickle

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

system_type = 1  # 系统类型变量 # 1 windows  2 ubuntu 3 mac


# 保存模型
def save_model(agent, filepath):
    torch.save(agent.model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


# 加载模型
def load_model(agent, filepath):
    if os.path.exists(filepath):
        agent.model.load_state_dict(torch.load(filepath))
        agent.model.eval()  # 设置为评估模式
        print(f"Model loaded from {filepath}")
    else:
        print(f"Model file {filepath} does not exist. Training from scratch.")


# 保存经验池
def save_replay_buffer(agent, filepath):
    """
    保存智能体的经验回放池到指定文件。
    """
    with open(filepath, 'wb') as f:
        pickle.dump(list(agent.memory), f)  # 将 deque 转换为列表进行保存
    print(f"Replay buffer saved to {filepath}")


# 加载经验池
def load_replay_buffer(agent, filepath):
    """
    从指定文件加载经验回放池。如果文件不存在，将输出警告。
    """
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            loaded_memory = pickle.load(f)
        agent.memory = deque(loaded_memory, maxlen=agent.memory.maxlen)  # 恢复为 deque
        print(f"Replay buffer loaded from {filepath}")
    else:
        print(f"Replay buffer file {filepath} does not exist. Training from scratch.")


def save_results_to_json(file_path, episode_number, actions, rewards, end_effector_positions, target_position):
    """
    保存运行结果到 JSON 文件。
    :param file_path: JSON 文件的路径。
    :param episode_number: 当前运行次数。
    :param actions: 动作列表。
    :param rewards: 奖励列表。
    :param end_effector_positions: 末端轨迹列表。
    :param target_position: 目标点坐标，格式为 (x, z)。
    """
    # 组织要保存的数据

    '''
    result_data = {
        "Episode": episode_number,
        "Actions": actions,
        "Rewards": rewards,
        "EndEffectorPositions": end_effector_positions,
        "TargetPosition": {"x": target_position[0], "z": target_position[1]}
    }    
    '''

    total_reward = sum(rewards)

    result_data = {
        "Episode": episode_number,
        "Actions": actions,
        "Rewards": total_reward,
        "TargetPosition": {"x": target_position[0], "z": target_position[1]}
    }
    # 检查文件是否存在，存在则追加
    if os.path.exists(file_path):
        # 读取现有数据
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        # 如果文件不存在，创建空列表
        data = []

    # 将当前结果追加到数据中
    data.append(result_data)

    # 保存数据到 JSON 文件
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def visualize_two_rewards(all_rewards_1, all_rewards_2):
    """
    在一个图中绘制两个子图，分别显示两种奖励函数。
    """
    plt.close('all')  # 关闭所有当前打开的绘图窗口，释放资源

    plt.ion()  # 开启交互模式

    # 创建一个包含两个子图的画布
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].clear()
    axes[1].clear()

    # 绘制第一种奖励

    axes[0].plot(all_rewards_1, label=f"Episode")

    axes[0].set_title("Total Reward")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True)

    # 绘制第二种奖励
    axes[1].plot(all_rewards_2, label=f"Episode")
    axes[1].set_title("Sum Reward")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Reward")
    axes[1].legend()
    axes[1].grid(True)

    # 调整子图之间的间距并显示
    plt.tight_layout()
    plt.pause(0.1)  # 暂停 0.1 秒用于刷新图像
    plt.ioff()  # 关闭交互模式（可选）


# 每次调用这个函数时，图像会自动刷新


def calculate_path_length(positions):
    """ 计算路径长度 """
    total_length = 0.0
    for i in range(1, len(positions)):
        x1, z1 = positions[i - 1]
        x2, z2 = positions[i]
        distance = math.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)
        total_length += distance
    return total_length


def calculate_smoothness(positions):
    """ 计算路径的平滑度，通过累加转向角度 """
    total_angle_change = 0.0
    for i in range(2, len(positions)):
        x1, z1 = positions[i - 2]
        x2, z2 = positions[i - 1]
        x3, z3 = positions[i]

        # 计算两个相邻方向向量的角度
        v1 = [x2 - x1, z2 - z1]
        v2 = [x3 - x2, z3 - z2]

        # 向量点积与模长
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        norm_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        norm_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if norm_v1 > 0 and norm_v2 > 0:
            cos_theta = dot_product / (norm_v1 * norm_v2)
            cos_theta = max(-1.0, min(1.0, cos_theta))  # 防止浮点误差导致越界
            angle = math.acos(cos_theta)  # 计算角度
            total_angle_change += angle
    return total_angle_change


# DQN 网络定义
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 环境与强化学习训练类
class DQNAgent:
    def __init__(self, input_dim, output_dim):

        if system_type == 3:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # 使用MPS加速或CPU
            self.model = DQN(input_dim, output_dim).to(device)  # 使用选定的设备
            self.target_model = DQN(input_dim, output_dim).to(device)
        else:
            self.model = DQN(input_dim, output_dim).cuda()  # 使用GPU加速
            self.target_model = DQN(input_dim, output_dim).cuda()

        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 0.5  # epsilon-greedy策略
        self.epsilon_decay = 0.995  # 衰减率，每次调用动作选择后减少
        self.min_epsilon = 0.01  # 最小 epsilon 值

        self.update_target_every = 10  # 每10步更新target网络
        self.steps_done = 0
        self.zero_action_counter = 0  # 计数器，记录当前需要强制执行的连续零动作次数
        self.required_zero_steps = 5  # 非零动作之后需要执行的连续零动作次数
        self.action_space_size = 41  # -20 到 20，总共 41 个动作
        self.training_start = 300  # 等待300步之后开始学习

    def select_action(self, state, episode):
        """
        选择动作，根据模型预测的 Q 值来选择最优动作。
        """
        # 1. 检查状态合法性
        if np.isnan(state).any() or np.isinf(state).any():
            print("Error: Invalid state data detected:", state)
            state = np.zeros_like(state)  # 如果状态不合法，重置为全零
            print("State reset to zeros.")

        # 2. 强制执行零动作机制
        if self.zero_action_counter > 0:
            self.zero_action_counter -= 1
            return 0  # 强制执行零动作

        # 3. 动作空间映射
        # 动作空间 [-20, -19, ..., 0, ..., 19, 20]
        action_space = list(range(-20, 21, 4))
        # action_space = list(range(-20, 1, 4)) + list(range(4, 21, 4))

        # 4. 状态转换为 GPU 张量
        if system_type == 3:
            device = next(self.model.parameters()).device  # 获取模型所在设备
            state = torch.FloatTensor(state).unsqueeze(0).to(device)  # 增加 batch 维度
        else:
            state = torch.FloatTensor(state).unsqueeze(0).cuda()  # 增加 batch 维度

        # 5. 模型预测 Q 值
        with torch.no_grad():
            q_values = self.model(state)  # 获取 Q 值，shape = [1, action_space_size]


        # 6. 更新 epsilon 值，随着训练次数增加逐渐减少随机性
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


        if random.random() < self.epsilon:  # 以概率 epsilon 随机探索
            action = random.choice(action_space)
        else:
            # 6. 选择最优动作索引
            best_action_index = q_values.argmax(dim=1).item()  # 获取 Q 值最大的索引
            # 7. 将索引映射到实际动作
            best_action_index = max(0, min(best_action_index, len(action_space) - 1))
            action = action_space[best_action_index]

        # 8. 零动作判断
        if action == 0:
            return action
        # 9. 启动零动作计数器，并返回非零动作
        self.zero_action_counter = self.required_zero_steps
        return action

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        # if len(self.memory) < self.batch_size:
        #    return

        if len(self.memory) < self.batch_size or self.steps_done < self.training_start:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 检查数据合法性
        if any(np.isnan(s).any() or np.isinf(s).any() for s in states + next_states):
            print("Error: Invalid state detected in replay buffer!")
            return

        if any(a < 0 or a >= self.action_space_size for a in actions):
            print("Error: Invalid action in replay buffer:", actions)
            return

        if system_type == 3:
            device = next(self.model.parameters()).device  # 获取模型所在设备
            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)
        else:
            states = torch.FloatTensor(states).cuda()
            actions = torch.LongTensor(actions).cuda()
            rewards = torch.FloatTensor(rewards).cuda()
            next_states = torch.FloatTensor(next_states).cuda()
            dones = torch.FloatTensor(dones).cuda()

        # 添加断言，检查索引是否超出范围
        assert actions.max() < self.action_space_size, f"Invalid action index {actions.max()} exceeds action_space_size {self.action_space_size}"
        assert actions.min() >= 0, f"Invalid action index {actions.min()} is less than 0"

        q_values = self.model(states).gather(1, actions.unsqueeze(1))

        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每隔一定步骤更新target网络
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())


# 处理接收 Unity 返回的数据
def receive_data(client_socket, data_event, received_message, stop_event):
    """
    接收 Unity 发送的数据并解析为状态变量（以逗号分隔的数值）。
    """
    while not stop_event.is_set():
        try:
            # 接收数据并清除首尾空格
            data = client_socket.recv(1024).decode('utf-8').strip()
            if data:
                # print(f"Received from Unity: {data}")
                try:
                    # 将字符串解析为浮点数列表
                    parsed_data = [float(x) for x in data.split(",")]
                    received_message[0] = parsed_data  # 更新接收到的数据
                except ValueError as e:
                    # print(f"Error parsing data '{data}': {e}. Skipping this data.")
                    received_message[0] = None  # 设置为无效数据

                data_event.set()  # 设置事件，通知主线程
            else:
                print("Warning: Received empty data.")
        except ConnectionResetError:
            print("Connection lost.")
            break
        except Exception as e:
            print(f"Error in receive_data: {e}")


def get_total_reward(end_effector_positions, parsed_data):
    end_x, end_z = end_effector_positions[-1]  # 路径末端的坐标
    if parsed_data is None:
        target_x, target_z = -0.09, 3.9
    else:
        target_x, target_z = parsed_data[-2], parsed_data[-1]  # 最后的目标点坐标
    distance_to_target = ((target_x - end_x) ** 2 + (target_z - end_z) ** 2) ** 0.5

    # 计算路径长度
    path_length = 0.0
    for i in range(1, len(end_effector_positions)):
        x1, z1 = end_effector_positions[i - 1]
        x2, z2 = end_effector_positions[i]
        path_length += ((x2 - x1) ** 2 + (z2 - z1) ** 2) ** 0.5

    # 计算路径平滑性
    path_smoothness_penalty = 0.0
    for i in range(2, len(end_effector_positions)):
        x1, z1 = end_effector_positions[i - 2]
        x2, z2 = end_effector_positions[i - 1]
        x3, z3 = end_effector_positions[i]
        # 计算两段的方向向量
        v1 = (x2 - x1, z2 - z1)
        v2 = (x3 - x2, z3 - z2)
        # 计算两段之间的夹角的余弦值
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        norm_v1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        norm_v2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
        cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-6)  # 避免除零

        # 角度变化的惩罚（1 - cos_theta 越大表示变化越大）
        path_smoothness_penalty += (1 - cos_theta)

        # 定义权重
        w_distance = 10.0  # 距离的权重
        w_length = 1.0  # 路径长度的权重
        w_smoothness = 5.0  # 平滑度的权重
        # 计算总奖励
        total_reward = (
                -w_distance * distance_to_target
                - w_length * path_length
                - w_smoothness * path_smoothness_penalty
        )

        return total_reward


def get_target_angle(rotation_y, target_z, position_z, target_x, position_x):
    theta_radians = math.radians(rotation_y)
    x1, y1 = math.cos(theta_radians), math.sin(theta_radians)
    x2, y2 = target_z - position_z, target_x - position_x
    # 计算两条线段模长
    magnitude1 = math.sqrt(x1 ** 2 + y1 ** 2)  # 第一条线段的模长（始终为1）
    magnitude2 = math.sqrt(x2 ** 2 + y2 ** 2)  # 第二条线段的模长
    # 计算点积和叉积
    dot_product = x1 * x2 + y1 * y2
    cross_product = x1 * y2 - y1 * x2
    # 计算夹角的余弦值
    cos_theta = dot_product / (magnitude1 * magnitude2)
    cos_theta = max(-1.0, min(1.0, cos_theta))  # 防止浮点数误差
    # 计算夹角的弧度值
    angle_radians = math.acos(cos_theta)
    # 判断夹角的正负（通过叉积符号）
    if cross_product < 0:
        angle_radians = -angle_radians  # 如果叉积为负，夹角为负
    angle_radians = math.degrees(angle_radians)
    #print(f'current dir angle:{angle_radians}')
    return angle_radians


def simulate_one_episode(client_socket, agent, data_event, received_message, input_sequence):
    # 初始化记录路径的列表
    end_effector_positions = []
    input_index = 0  # 用于跟踪当前发送位置
    # 发送开始信号
    start_signal = "START"
    client_socket.send(start_signal.encode('utf-8'))
    # print(f"Sent to Unity: {start_signal}")
    time.sleep(0.05)  # 等待 50 毫秒
    state = [0, 0, 0, -0.09, 3.9]  # 初始状态，假设为零状态
    next_state = [0, 0, 0, -0.09, 3.9]
    total_reward = 0

    # 初始化动作列表和
    action_list = []
    reward_list = []

    while input_index < len(input_sequence):  # 发送序列值
        distance, angle_deg = 0, 0

        if input_index > 0:

            # print("Waiting for Unity's response...")
            data_event.wait()  # 等待接收到 Unity 的消息
            # print(f"Received message: {received_message[0]}")

            parsed_data = received_message[0]
            data_event.clear()  # 重置事件

            # 提取状态变量和目标点
            # position_x, position_z, rotation_y, target_x, target_z = parsed_data
            if parsed_data is not None:
                position_x, position_z, rotation_y, target_x, target_z = parsed_data
            else:
                # 处理 parsed_data 为 None 的情况，可能设置默认值或者报错
                print("Error: parsed_data is None.")
                # 或者可以给出一个默认值:
                position_x, position_z, rotation_y, target_x, target_z = 0, 0, 0, 0.06, 3.876

            next_state = [position_x, position_z, rotation_y, target_x, target_z]  # 状态变量
            # 记录末端坐标
            end_effector_positions.append((position_x, position_z))

            # 计算欧几里得距离
            distance = ((target_x - position_x) ** 2 + (target_z - position_z) ** 2) ** 0.5
            # reward = -distance  # 距离越小，奖励越高
            path_length = calculate_path_length(end_effector_positions)

            angle_deg = get_target_angle(rotation_y, target_z, position_z, target_x, position_x)


            # reward = -distance - path_length

            reward = -distance

            # 添加目标点附近的增益
            if 0.2 > distance > 0.1:  # 假设0.5是靠近目标点的阈值
                reward += 20  # 额外奖励
            elif distance < 0.1:
                reward = reward + 50



        else:
            state = [0, 0, 0, -0.09, 3.9]  # 初始状态
            reward = 0.0
            action = 0  # 初始动作为零

        # 选择动作
        action = agent.select_action(state)
        client_socket.send(str(action).encode('utf-8'))
        # print(f"Sent action {action} to Unity")

        penalty_for_turning = 0.1  # 非零动作的固定惩罚值 狠狠的惩罚转向
        reward -= penalty_for_turning * (abs(action) ** 1.5)
        total_reward = total_reward + reward

        action_list.append(action)
        reward_list.append(reward)

        # **新增判断**：如果距离小于 0.1，停止当前回合
        if input_index > 0:
            if distance < 0.1 or (abs(angle_deg) and distance > 0.4) > 90:
                if distance < 0.1:
                    print("Reached the target! Ending current episode.")
                elif abs(angle_deg) > 90 and distance > 0.4:
                    print("CAN NOT Reached the target! Ending current episode.")
                    reward = reward - 15

                agent.store_experience(state, action + 20, reward, next_state, done=False)
                agent.learn()
                break  # 提前结束循环
        # 存储经验并学习


        # 更新 state 为 next_state
        state = next_state

        # 等待 50 毫秒后再发送下一个数据
        time.sleep(0.05)  # 延迟 50 毫秒
        input_index += 1

    # 发送结束信号
    end_signal = "END"
    client_socket.send(end_signal.encode('utf-8'))
    # print(f"Sent to Unity: {end_signal}")
    time.sleep(0.08)  # 等待 80 毫秒

    return end_effector_positions, parsed_data, total_reward, action_list, reward_list


# 启动客户端连接到 Unity
def start_client(host='127.0.0.1', port=9999, num_simulations=5, load_previous_model=False):
    # 定义输入序列
    input_sequence = np.zeros(38)  # 例：输入序列

    # 全局变量用于存储所有仿真的总奖励
    all_rewards1 = []
    all_rewards2 = []
    average_reward1 = []
    average_reward2 = []

    # 线程同步事件
    data_event = threading.Event()
    stop_event = threading.Event()  # 用于停止接收线程
    received_message = [None]  # 用于存储接收到的数据

    # 创建 Socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print("Connected to Unity")

    # 启动接收线程
    receive_thread = threading.Thread(
        target=receive_data, args=(client_socket, data_event, received_message, stop_event), daemon=True
    )
    receive_thread.start()

    # 初始化 DQN agent
    agent = DQNAgent(input_dim=5, output_dim=11)  # 假设状态维度为 3，动作维度为 2

    # 尝试加载模型和经验池
    if system_type == 1:
        model_path = "C:\\Users\\ayaka\\Desktop\\dqn_model.pth"
        replay_buffer_path = "C:\\Users\\ayaka\\Desktop\\replay_buffer.pkl"
    elif system_type == 2:
        model_path = "/media/dockang/D/haoran/pythonProject1/dqn_model.pth"
        replay_buffer_path = "/media/dockang/D/haoran/pythonProject1/replay_buffer.pkl"
    elif system_type == 3:
        model_path = "/Users/ayaka/Desktop/pythonProject1/dqn_model.pth"
        replay_buffer_path = "/Users/ayaka/Desktop/pythonProject1/replay_buffer.pkl"

    if load_previous_model:
        load_model(agent, model_path)
        load_replay_buffer(agent, replay_buffer_path)

    try:
        for sim in range(num_simulations):
            print(f"Starting simulation {sim + 1} of {num_simulations}...\n")

            # 进行一次仿真
            end_effector_positions, parsed_data, total_reward1, action_list, reward_list = simulate_one_episode(
                client_socket, agent, data_event,
                received_message, input_sequence)

            if parsed_data is None:
                target_position = np.array([0.06, 3.876])
            else:
                target_position = parsed_data[-2:]

            # 得到总奖励
            total_reward = get_total_reward(end_effector_positions, parsed_data)

            print(f"Simulation {sim + 1} complete.\n")

            all_rewards1.append(total_reward1)
            all_rewards2.append(total_reward1)
            average_reward1.append(np.mean(all_rewards1))
            average_reward2.append(np.mean(all_rewards2))

            visualize_two_rewards(average_reward1, average_reward2)

            if system_type == 1:
                file_path = "C:\\Users\\ayaka\\Desktop\\sim_data.json"
            elif system_type == 2:
                file_path = "/media/dockang/D/haoran/pythonProject1/sim_data.json"
            elif system_type == 3:
                file_path = "/Users/ayaka/Desktop/pythonProject1/sim_data.json"

            '''
            if (sim+1) % 5 == 0:
                stop_command = "STOP0"
                client_socket.send(stop_command.encode('utf-8'))
                data_event.wait()  # 等待事件触发
                # 解析收到的消息
                parsed_data = received_message[0]
                if parsed_data == "RESET_COMPLETE":
                    # 重置事件和消息
                    data_event.clear()
                    received_message[0] = None
                    time.sleep(5)            
            '''



            save_results_to_json(file_path, sim, action_list, reward_list, end_effector_positions, target_position)
            time.sleep(2)  # 每次仿真结束后等待 1 秒钟

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # 保存模型和经验池
        save_model(agent, model_path)
        save_replay_buffer(agent, replay_buffer_path)

        # 设置停止事件，通知接收线程退出
        stop_event.set()
        client_socket.close()
        receive_thread.join()  # 等待接收线程结束
        print("Connection closed.")


if __name__ == "__main__":
    start_time = time.time()

    start_client(num_simulations=1000, load_previous_model=False)  # 设置仿真次数为 5

    # 记录程序结束时间
    end_time = time.time()
    # 计算程序运行时长
    elapsed_time = end_time - start_time
    print(f"Program ran for: {elapsed_time:.2f} seconds")
