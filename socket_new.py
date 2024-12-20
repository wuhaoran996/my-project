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
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def calculate_path_length(positions):
    """ 计算路径长度 """
    total_length = 0.0
    for i in range(1, len(positions)):
        x1, z1 = positions[i-1]
        x2, z2 = positions[i]
        distance = math.sqrt((x2 - x1)**2 + (z2 - z1)**2)
        total_length += distance
    return total_length

def calculate_smoothness(positions):
    """ 计算路径的平滑度，通过累加转向角度 """
    total_angle_change = 0.0
    for i in range(2, len(positions)):
        x1, z1 = positions[i-2]
        x2, z2 = positions[i-1]
        x3, z3 = positions[i]

        # 计算两个相邻方向向量的角度
        v1 = [x2 - x1, z2 - z1]
        v2 = [x3 - x2, z3 - z2]

        # 向量点积与模长
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        norm_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        norm_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

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
        self.model = DQN(input_dim, output_dim).cuda()  # 使用GPU加速
        self.target_model = DQN(input_dim, output_dim).cuda()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 0.1  # epsilon-greedy策略
        self.update_target_every = 10  # 每10步更新target网络
        self.steps_done = 0
        self.zero_action_counter = 0  # 计数器，记录当前需要强制执行的连续零动作次数
        self.required_zero_steps = 5  # 非零动作之后需要执行的连续零动作次数
        self.action_space_size = 41  # -20 到 20，总共 41 个动作

    '''
        def select_action(self, state):
            # 如果处于零动作约束阶段
            if self.zero_action_counter > 0:
                self.zero_action_counter -= 1
                return 0  # 强制选择 0 动作
    
            # 随机以 50% 概率选择 0 或非 0 动作
            if random.random() < 0.5:
                self.zero_action_counter = 5  # 设置零动作约束阶段
                return 0  # 选择 0 动作
            else:
                # 从模型预测中选择非零动作
                state = torch.FloatTensor(state).unsqueeze(0).cuda()
                with torch.no_grad():
                    q_values = self.model(state)  # 获取当前状态下的 Q 值
                # 将非零动作的 Q 值提取出来
                non_zero_actions = [x for x in range(-20, 21) if x != 0]
                q_values_non_zero = q_values.squeeze()[non_zero_actions]  # 提取非零动作的 Q 值
                action = non_zero_actions[q_values_non_zero.argmax().item()]  # 选择 Q 值最大的非零动作
    
                # 设置零动作约束
                if action != 0:
                    self.zero_action_counter = 5  # 触发连续 5 次零动作的约束
                return action
    '''

    '''
    def select_action(self, state):
        # 检查状态合法性
        if np.isnan(state).any() or np.isinf(state).any():
            print("Error: Invalid state data detected:", state)
            state = np.zeros_like(state)  # 重置状态

        # 如果需要强制执行零动作
        if self.zero_action_counter > 0:
            self.zero_action_counter -= 1
            return 0  # 强制执行零动作
        # 计算模型预测的动作
        state = torch.FloatTensor(state).unsqueeze(0).cuda()  # 将状态转换为张量并移至 GPU

        with torch.no_grad():
            q_values = self.model(state)  # 使用模型预测 Q 值

        # 将动作空间范围从 -20 到 20
        action_space = list(range(-20, 21))  # 包括 -20 到 20 的整数

        # 根据 Q 值选择动作
        best_action_index = q_values.argmax().item()

        action = action_space[best_action_index % len(action_space)]  # 选择动作，循环取模防止越界

        # 如果选出的动作是零动作，直接返回
        if action == 0:
            return action

        # 如果选出的是非零动作，启动零动作计数器，并返回该动作
        self.zero_action_counter = self.required_zero_steps
        return action    
    
    '''

    def select_action(self, state):
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
        action_space = list(range(-20, 21))  # 动作空间 [-20, -19, ..., 0, ..., 19, 20]

        # 4. 状态转换为 GPU 张量
        state = torch.FloatTensor(state).unsqueeze(0).cuda()  # 增加 batch 维度

        # 5. 模型预测 Q 值
        with torch.no_grad():
            q_values = self.model(state)  # 获取 Q 值，shape = [1, action_space_size]

        # 6. 选择最优动作索引
        best_action_index = q_values.argmax(dim=1).item()  # 获取 Q 值最大的索引

        # 7. 将索引映射到实际动作
        if best_action_index < 0 or best_action_index >= len(action_space):
            print(f"Warning: Predicted action index {best_action_index} out of bounds.")
            best_action_index = 0  # 索引越界时，默认为第一个动作

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
        if len(self.memory) < self.batch_size:
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

        states = torch.FloatTensor(states).cuda()
        actions = torch.LongTensor(actions).cuda()
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = torch.FloatTensor(next_states).cuda()
        dones = torch.FloatTensor(dones).cuda()

        #print(f"Actions: {actions}")  # 检查动作索引
        #print(f"Q-values shape: {self.model(states).shape}")  # 确保 Q 值输出维度
        #print(f"Actions shape: {actions.shape}")
        #print(f"Actions max: {actions.max()}, Actions min: {actions.min()}")

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
                    print(f"Error parsing data '{data}': {e}. Skipping this data.")
                    received_message[0] = None  # 设置为无效数据

                data_event.set()  # 设置事件，通知主线程
            else:
                print("Warning: Received empty data.")
        except ConnectionResetError:
            print("Connection lost.")
            break
        except Exception as e:
            print(f"Error in receive_data: {e}")


def get_total_reward(end_effector_positions,parsed_data):
    end_x, end_z = end_effector_positions[-1]  # 路径末端的坐标
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
        w_distance = 10.0   # 距离的权重
        w_length = 1.0      # 路径长度的权重
        w_smoothness = 5.0  # 平滑度的权重
        # 计算总奖励
        total_reward = (
            -w_distance * distance_to_target
            - w_length * path_length
            - w_smoothness * path_smoothness_penalty
        )

        return total_reward

def simulate_one_episode(client_socket, agent, data_event, received_message,input_sequence):
    # 初始化记录路径的列表
    end_effector_positions = []
    input_index = 0  # 用于跟踪当前发送位置
    # 发送开始信号
    start_signal = "START"
    client_socket.send(start_signal.encode('utf-8'))
    print(f"Sent to Unity: {start_signal}")
    time.sleep(0.05)  # 等待 50 毫秒
    state = [0, 0, 0]  # 初始状态，假设为零状态
    next_state = [0, 0, 0]

    while input_index < len(input_sequence):  # 发送序列值
        if input_index > 0:
            # print("Waiting for Unity's response...")
            data_event.wait()  # 等待接收到 Unity 的消息
            print(f"Received message: {received_message[0]}")

            parsed_data = received_message[0]
            data_event.clear()  # 重置事件

            # 提取状态变量和目标点
            position_x, position_z, rotation_y, target_x, target_z = parsed_data
            next_state = [position_x, position_z, rotation_y]  # 状态变量
            # 记录末端坐标
            end_effector_positions.append((position_x, position_z))

            # 计算欧几里得距离
            distance = ((target_x - position_x) ** 2 + (target_z - position_z) ** 2) ** 0.5
            reward = -distance  # 距离越小，奖励越高
            # 添加目标点附近的增益
            if distance < 0.2:  # 假设0.5是靠近目标点的阈值
                reward += 10.0  # 额外奖励

        else:
            state = [0, 0, 0]  # 初始状态
            reward = 0.0
            action = 0  # 初始动作为零

        # 选择动作
        action = agent.select_action(state)
        client_socket.send(str(action).encode('utf-8'))
        print(f"Sent action {action} to Unity")

        penalty_for_turning = 1.0  # 非零动作的固定惩罚值
        reward -= penalty_for_turning * abs(action)

        # 存储经验并学习
        agent.store_experience(state, action + 20, reward, next_state, done=False)
        agent.learn()

        # 更新 state 为 next_state
        state = next_state

        # 等待 50 毫秒后再发送下一个数据
        time.sleep(0.05)  # 延迟 50 毫秒
        input_index += 1

    # 发送结束信号
    end_signal = "END"
    client_socket.send(end_signal.encode('utf-8'))
    print(f"Sent to Unity: {end_signal}")
    time.sleep(0.08)  # 等待 80 毫秒

    return end_effector_positions, parsed_data



# 启动客户端连接到 Unity
def start_client(host='127.0.0.1', port=9999, num_simulations=5):
    # 定义输入序列
    input_sequence = np.zeros(30)  # 例：输入序列

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
    agent = DQNAgent(input_dim=3, output_dim=40)  # 假设状态维度为 3，动作维度为 2

    try:
        for sim in range(num_simulations):
            print(f"Starting simulation {sim + 1} of {num_simulations}...\n")

            '''
            # 初始化记录路径的列表
            end_effector_positions = []
            input_index = 0  # 用于跟踪当前发送位置
            # 发送开始信号
            start_signal = "START"
            client_socket.send(start_signal.encode('utf-8'))
            print(f"Sent to Unity: {start_signal}")
            time.sleep(0.05)  # 等待 50 毫秒

            state = [0, 0, 0]  # 初始状态，假设为零状态
            next_state=[0, 0, 0]

            while input_index < len(input_sequence):  # 发送序列值
                if input_index > 0:
                    # print("Waiting for Unity's response...")
                    data_event.wait()  # 等待接收到 Unity 的消息
                    print(f"Received message: {received_message[0]}")

                    parsed_data = received_message[0]
                    data_event.clear()  # 重置事件

                    # 提取状态变量和目标点
                    position_x, position_z, rotation_y, target_x, target_z = parsed_data
                    next_state  = [position_x, position_z, rotation_y]  # 状态变量
                    # 记录末端坐标
                    end_effector_positions.append((position_x, position_z))

                    # 计算欧几里得距离
                    distance = ((target_x - position_x) ** 2 + (target_z - position_z) ** 2) ** 0.5
                    reward = -distance  # 距离越小，奖励越高
                    # 添加目标点附近的增益
                    if distance < 0.2:  # 假设0.5是靠近目标点的阈值
                        reward += 10.0  # 额外奖励

                else:
                    state = [0, 0, 0]  # 初始状态
                    reward = 0.0
                    action = 0  # 初始动作为零

                # 选择动作
                action = agent.select_action(state)
                client_socket.send(str(action).encode('utf-8'))
                print(f"Sent action {action} to Unity")

                penalty_for_turning = 1.0  # 非零动作的固定惩罚值
                reward -= penalty_for_turning * abs(action)

                # 存储经验并学习
                agent.store_experience(state, action+20, reward, next_state , done=False)
                agent.learn()

                # 更新 state 为 next_state
                state = next_state

                # 等待 50 毫秒后再发送下一个数据
                time.sleep(0.05)  # 延迟 50 毫秒
                input_index += 1

            # 发送结束信号
            end_signal = "END"
            client_socket.send(end_signal.encode('utf-8'))
            print(f"Sent to Unity: {end_signal}")
            time.sleep(0.08)  # 等待 80 毫秒            
            '''

            # 进行一次仿真
            end_effector_positions, parsed_data = simulate_one_episode(client_socket, agent, data_event, received_message, input_sequence)

            #得到总奖励
            get_total_reward(end_effector_positions, parsed_data)

            print(f"Simulation {sim + 1} complete.\n")
            time.sleep(1)  # 每次仿真结束后等待 1 秒钟

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # 设置停止事件，通知接收线程退出
        stop_event.set()
        client_socket.close()
        receive_thread.join()  # 等待接收线程结束
        print("Connection closed.")



if __name__ == "__main__":
    start_client(num_simulations=10)  # 设置仿真次数为 5
