import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from rdp import rdp  # Ramer-Douglas-Peucker库

import os
import subprocess
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# 选择设备：如果有可用的GPU，使用GPU；否则，使用CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 超参数定义
STATE_SIZE = 20       # 假设轨迹特征编码为20维向量
ACTION_SIZE = 3       # 动作参数：初始角度、预弯曲角度、预弯曲位置
GAMMA = 0.99          # 折扣因子
LR = 0.001            # 学习率
BATCH_SIZE = 64       # 批量大小
MEMORY_SIZE = 10000   # 经验回放池大小
EPSILON = 1.0         # 初始探索率
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10  # 每多少轮更新目标网络

def extract_state_vector(robot_path_file, env_file, max_segments=6, max_obstacles=2):
    """
    提取机器人路径的特征点和环境障碍物信息，生成固定维度的一维状态向量。

    :param robot_path_file: 机器人路径数据的 JSON 文件路径
    :param env_file: 环境障碍物数据的 JSON 文件路径
    :param max_segments: 轨迹的最大特征点数（不足补 0）
    :param max_obstacles: 障碍物的最大数量（不足补 0）
    :return: 固定维度的一维状态向量
    """
    # 读取机器人路径数据
    with open(robot_path_file, "r") as file:
        robot_data = json.load(file)

    # 读取环境障碍物数据
    with open(env_file, "r") as file:
        env_data = json.load(file)

    # 提取机器人路径
    robot_positions = [
        (entry["EndEffectorPosition"]["x"], entry["EndEffectorPosition"]["z"])
        for entry in robot_data.get("RobotShapeHistory", [])
    ]

    # 使用 Ramer-Douglas-Peucker 算法简化轨迹
    simplified_positions = rdp(robot_positions, epsilon=0.03)

    # 如果特征点不足 max_segments，补 0；如果超过 max_segments，截断
    simplified_positions = np.array(simplified_positions[:max_segments])
    if simplified_positions.shape[0] < max_segments:
        padding = np.zeros((max_segments - simplified_positions.shape[0], 2))
        simplified_positions = np.vstack((simplified_positions, padding))

    # 提取环境障碍物信息
    obstacles = env_data.get("Cubes", [])
    obstacle_features = []
    for obstacle in obstacles[:max_obstacles]:
        position = obstacle.get("Position", {})
        scale = obstacle.get("Scale", {})
        rotation = obstacle.get("Rotation", {})

        # 提取中心点、宽、高和旋转角度
        x = position.get("x", 0)
        z = position.get("z", 0)
        width = scale.get("x", 0)
        height = scale.get("z", 0)
        angle = np.arctan2(rotation.get("y", 0), rotation.get("w", 1)) * 2  # 从四元数计算旋转角

        obstacle_features.append([x, z, width, height, angle])

    # 如果障碍物不足 max_obstacles，补 0；如果超过 max_obstacles，截断
    obstacle_features = np.array(obstacle_features[:max_obstacles])
    if obstacle_features.shape[0] < max_obstacles:
        padding = np.zeros((max_obstacles - obstacle_features.shape[0], 5))
        obstacle_features = np.vstack((obstacle_features, padding))

    # 将轨迹特征点和障碍物特征展平为一维状态向量
    state_vector = np.hstack((simplified_positions.flatten(), obstacle_features.flatten()))
    return state_vector


def analyze_robot_path1(robot_path_file, env_file, max_segments, max_obstacles, epsilon=0.03):
    """
    从JSON文件中提取轨迹和环境信息，并生成固定维度的状态变量。

    参数:
    - robot_path_file: 机器人路径JSON文件路径。
    - env_file: 环境信息JSON文件路径。
    - max_segments: 最大轨迹段数。
    - max_obstacles: 最大障碍物数量。
    - epsilon: RDP算法的简化阈值。

    返回:
    - state_vector: 包含轨迹特征、障碍物特征、目标点和预弯曲点的固定维度状态向量。
    """
    # 读取机器人路径数据
    with open(robot_path_file, "r") as file:
        robot_data = json.load(file)

    # 读取环境障碍物数据
    with open(env_file, "r") as file:
        env_data = json.load(file)

    # 提取机器人路径
    robot_path = robot_data.get("RobotShapeHistory", [])
    robot_positions = [
        (entry["EndEffectorPosition"]["x"], entry["EndEffectorPosition"]["z"])
        for entry in robot_path
    ]

    # 使用 RDP 算法简化轨迹
    simplified_positions = rdp(robot_positions, epsilon=epsilon)

    print(simplified_positions)

    # 固定轨迹段数
    fixed_trajectory = fix_path_segments(simplified_positions, max_segments)

    # 提取障碍物信息
    obstacles = env_data.get("Cubes", [])
    obstacle_features = extract_obstacle_features(obstacles, max_obstacles)

    state_vector =[fixed_trajectory,obstacle_features]

    return state_vector

def fix_path_segments(simplified_path, fixed_segment_count):
    """
    固定轨迹段数，使用截断或插值保证轨迹点数为 fixed_segment_count + 1。

    参数:
    - simplified_path: 简化后的路径点 [(x1, z1), ...]
    - fixed_segment_count: 固定的轨迹段数。

    返回:
    - fixed_path: 调整后的固定路径点 [(x1, z1), ...]
    """
    path_array = np.array(simplified_path)
    num_points = len(path_array)

    if num_points > fixed_segment_count + 1:
        # 超过固定段数，截断
        indices = np.linspace(0, num_points - 1, fixed_segment_count + 1, dtype=int)
        fixed_path = path_array[indices]
    elif num_points < fixed_segment_count + 1:
        # 不足固定段数，插值补齐
        t = np.linspace(0, 1, num_points)
        t_fixed = np.linspace(0, 1, fixed_segment_count + 1)
        fixed_path = np.array([
            np.interp(t_fixed, t, path_array[:, dim]) for dim in range(path_array.shape[1])
        ]).T
    else:
        # 恰好匹配固定段数
        fixed_path = path_array

    return fixed_path.tolist()

def extract_obstacle_features(obstacles, max_obstacles):
    """
    提取障碍物的特征（中心点 + 宽高 + 旋转角度）。

    参数:
    - obstacles: 障碍物列表 [{"Position": {...}, "Scale": {...}, "Rotation": {...}}, ...]
    - max_obstacles: 最大障碍物数量。

    返回:
    - features: 固定维度的障碍物特征。
    """
    obstacle_features = []
    for obstacle in obstacles[:max_obstacles]:
        position = obstacle.get("Position", {})
        scale = obstacle.get("Scale", {})
        rotation = obstacle.get("Rotation", {})

        center_x = position.get("x", 0)
        center_z = position.get("z", 0)
        width = scale.get("x", 0)
        height = scale.get("z", 0)
        rotation_angle = np.arctan2(rotation.get("y", 0), rotation.get("w", 1)) * 2

        obstacle_features.extend([center_x, center_z, width, height, rotation_angle])

    # 如果障碍物数量不足，填充 0
    if len(obstacles) < max_obstacles:
        obstacle_features.extend([0] * 5 * (max_obstacles - len(obstacles)))

    return obstacle_features




def analyze_robot_path(robot_path_file, env_file, visualize=False):

    # 读取机器人路径数据
    #robot_path_file = r"C:\\Users\\ayaka\\Desktop\\2024-12-14_15-09-00.json"
    with open(robot_path_file, "r") as file:
        robot_data = json.load(file)

    # 读取环境障碍物数据
    #obstacles_file = r"C:\\Users\\ayaka\\Desktop\\Env_0_20241214_150901.json"
    with open(env_file, "r") as file:
        env_data = json.load(file)

    # 提取机器人路径
    robot_path_file  = robot_data.get("RobotShapeHistory", [])
    robot_positions = [
        (entry["EndEffectorPosition"]["x"], entry["EndEffectorPosition"]["z"])
        for entry in robot_path_file
    ]

    # 使用 Ramer-Douglas-Peucker 算法简化轨迹
    simplified_positions = rdp(robot_positions, epsilon=0.03)

    '''
    # 生成线段信息
    segments = []
    for i in range(len(simplified_positions) - 1):
        start = simplified_positions[i]
        end = simplified_positions[i + 1]
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)
        direction_normalized = direction / (length if length > 0 else 1)  # 归一化方向向量
        segments.append({
            "start": start.tolist(),
            "end": end.tolist(),
            "length": length,
            "direction": direction_normalized.tolist()
        })    
    '''


    # 提取环境障碍物数据
    cubes = env_data.get("Cubes", [])

    # 提取 prebend_pos
    prebend_pos = env_data.get("prebend_pos", {})
    prebend_x = prebend_pos.get("x", None)
    prebend_z = prebend_pos.get("z", None)

    target = env_data.get("Target", {})
    target_x = target.get("x", None)
    target_z = target.get("z", None)

    prebend_angle = env_data.get("PrebendAngle", None)

    # 路径长度计算
    robot_positions = np.array(robot_positions)
    path_length = np.sum(np.sqrt(np.sum(np.diff(robot_positions, axis=0)**2, axis=1)))


    # 路径平滑度计算（方向变化率）
    diffs = np.diff(robot_positions, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    angle_diffs = np.diff(angles)
    smoothness = np.std(angle_diffs)

    # 终点距离目标点
    end_x, end_z = robot_positions[-1]
    distance_to_target = np.sqrt((end_x - target_x)**2 + (end_z - target_z)**2)

    # 打印结果
    # print(f"路径长度: {path_length:.2f}")
    # print(f"路径平滑度: {smoothness:.4f}")
    # print(f"路径终点到目标点的距离: {distance_to_target:.2f}")

    if visualize:

        # 创建一个图形
        plt.figure(figsize=(10, 10))
        # 绘制机器人路径
        robot_positions = np.array(robot_positions)
        plt.plot(robot_positions[:, 0], robot_positions[:, 1], 'r-', label="Robot Path")  # 红线表示机器人路径
        plt.scatter(robot_positions[:, 0], robot_positions[:, 1], c='red', s=20, label="Robot Points")  # 用红点表示关键位置

        # 简化轨迹
        simplified_positions = np.array(simplified_positions)
        plt.plot(simplified_positions[:, 0], simplified_positions[:, 1], 'b--', label="Simplified Path")
        plt.scatter(simplified_positions[:, 0], simplified_positions[:, 1], c='blue', s=50, label="Key Points")


        # 绘制环境障碍物
        for cube in cubes:
            # 获取立方体的位置、旋转和缩放
            position = cube.get("Position", {})
            rotation = cube.get("Rotation", {})
            scale = cube.get("Scale", {})

            # 获取立方体的中心位置
            center = np.array([position.get("x", 0), position.get("z", 0)])
            # 获取立方体的缩放因子
            scale_x = abs(scale.get("x", 1))
            scale_z = abs(scale.get("z", 1))

            # 立方体的四个顶点（忽略 y 轴，只考虑 x 和 z 坐标）
            vertices = np.array([
                [-0.5, -0.5],  # 左下角
                [0.5, -0.5],   # 右下角
                [0.5, 0.5],    # 右上角
                [-0.5, 0.5]    # 左上角
            ]) * [scale_x, scale_z]  # 按照缩放因子调整

            # 将顶点平移到立方体的中心位置
            vertices += center

            # 旋转立方体（假设绕 Y 轴旋转）
            angle = np.arctan2(rotation.get("y", 0), rotation.get("w", 1)) * 2  # 从四元数计算旋转角度
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            rotated_vertices = np.dot(vertices - center, rotation_matrix) + center

            # 绘制立方体的边
            for i in range(4):
                plt.plot(
                    [rotated_vertices[i, 0], rotated_vertices[(i + 1) % 4, 0]],
                    [rotated_vertices[i, 1], rotated_vertices[(i + 1) % 4, 1]],
                    'b-'  # 蓝线表示障碍物
                )
            plt.fill(rotated_vertices[:, 0], rotated_vertices[:, 1], color='blue', alpha=0.3, label="Obstacle" if i == 0 else None)  # 填充障碍物

        # 标记预弯曲位置
        if prebend_x is not None and prebend_z is not None:
            plt.scatter(prebend_x, prebend_z, c='green', s=100, label="Prebend Pos", marker='x')
            plt.text(prebend_x, prebend_z, "Prebend Pos", color='green', fontsize=10, ha='right')

        # 标记目标点位置
        if target_x is not None and target_z is not None:
            plt.scatter(target_x, target_z, c='purple', s=100, label="Target Pos", marker='o')
            plt.text(target_x, target_z, "Target Pos", color='purple', fontsize=10, ha='right')

        # 设置图形属性
        plt.xlabel('X Coordinate')
        plt.ylabel('Z Coordinate')
        plt.title('Robot Path and Obstacles')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.grid()
        plt.show()

    # 返回计算结果
    return {
        "Length": path_length,
        "Smoothness": smoothness,
        "Distance": distance_to_target
    }


'''
def run_unity_scene_with_absolute_path(project_path, absolute_scene_path, unity_exe_path):
    """
    通过绝对路径运行 Unity 场景
    :param project_path: Unity 项目的根路径
    :param absolute_scene_path: 场景文件的绝对路径
    :param unity_exe_path: Unity Editor 的可执行文件路径
    """
    # 将绝对路径转换为 Unity 项目中的相对路径
    project_path = os.path.abspath(project_path)
    relative_scene_path = os.path.relpath(absolute_scene_path, start=project_path)

    # 确保路径分隔符为 Unity 可接受的格式
    relative_scene_path = relative_scene_path.replace(os.path.sep, '/')

    args = [
        unity_exe_path,
        "-projectPath", project_path,
        "-executeMethod", "SceneRunner.RunScene",
        "-scene", relative_scene_path
    ]

    try:
        subprocess.run(args, check=True)
        print(f"Scene '{relative_scene_path}' executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Scene execution failed: {e}")

'''

# 读取机器人路径数据
robot_path_file = r"C:\\Users\\ayaka\\Desktop\\2024-12-14_15-09-00.json"
# 读取环境障碍物数据
env_file = r"C:\\Users\\ayaka\\Desktop\\Env_0_20241214_150901.json"

state_vector = extract_state_vector(robot_path_file, env_file, max_segments=7, max_obstacles=2)


#state = analyze_robot_path1(robot_path_file, env_file, max_segments=6, max_obstacles=2, epsilon=0.03)
print("State Vector:", state_vector)


#print(len(state_vector))

#results = analyze_robot_path(robot_path_file, env_file, visualize=True)
#print("Length:", results["Length"])
#print("Smoothness:", results["Smoothness"])
#print("Distance:", results["Distance"])


'''
unity_exe_path = r"C:\\Program Files\\Tuanjie\\Hub\\Editor\\2022.3.48t1\\Editor\\Tuanjie.exe"
project_path = r"C:\\Users\\ayaka\\My project"
absolute_scene_path = r"C:\\Users\\ayaka\\My project\\Assets\\Scene1.scene"
run_unity_scene_with_absolute_path(project_path, absolute_scene_path, unity_exe_path)
'''




# 定义经验回放池
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    def __len__(self):
        return len(self.buffer)


# 定义轨迹特征提取函数
def extract_trajectory_features(trajectory):
    """
    输入轨迹点列表，返回特征向量
    :param trajectory: 轨迹点列表，例如 [[x1, z1], [x2, z2], ...]
    :return: 固定长度特征向量
    """
    trajectory = np.array(trajectory)
    if trajectory.shape[0] == 0:
        return np.zeros(STATE_SIZE)

    x = trajectory[:, 0]
    z = trajectory[:, 1]

    # 计算特征
    features = [
        np.mean(x), np.mean(z),  # 均值
        np.std(x), np.std(z),  # 标准差
        np.min(x), np.min(z),  # 最小值
        np.max(x), np.max(z),  # 最大值
        np.ptp(x), np.ptp(z)  # 极差 (max-min)
    ]
    features.extend(np.histogram(x, bins=5, range=(-10, 10))[0])  # 直方图特征
    features.extend(np.histogram(z, bins=5, range=(-10, 10))[0])  # 直方图特征

    return np.array(features[:STATE_SIZE])  # 截取固定维度

# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# DQN Agent
class DQNAgent:
    def __init__(self):
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.memory = ReplayBuffer(MEMORY_SIZE)

        self.model = DQN(STATE_SIZE, ACTION_SIZE).to(device)   # 训练模型
        self.target_model = DQN(STATE_SIZE, ACTION_SIZE).to(device)   # 目标模型
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

        self.epsilon = EPSILON

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-1, 1, self.action_size)  # 动作范围 [-1, 1]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.cpu().numpy()[0]  # 连续动作

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        current_q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        max_next_q_values = next_q_values.max(1)[0].detach()

        target_q_values = rewards + (1 - dones) * GAMMA * max_next_q_values

        predicted_q_values = current_q_values.gather(1, actions.argmax(dim=1, keepdim=True)).squeeze(1)

        loss = self.loss_fn(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# Unity与Python通信函数
def communicate_with_unity(file_path="robot_data.json"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        os.remove(file_path)
        return data
    return None

def send_action_to_unity(action, file_path="robot_action.json"):
    action_dict = {
        "initial_angle": action[0],
        "pre_bend_angle": action[1],
        "pre_bend_position": action[2]
    }
    with open(file_path, "w") as f:
        json.dump(action_dict, f)

# 主训练循环
if __name__ == "__main__":
    agent = DQNAgent()
    num_episodes = 500

    for e in range(num_episodes):
        state_data = communicate_with_unity()
        if state_data is None:
            print("等待Unity环境提供初始状态...")
            continue

        trajectory = state_data['trajectory']  # 假设Unity返回轨迹点列表
        state = extract_trajectory_features(trajectory)
        total_reward = 0
        done = False

        while not done:
            # 和环境交互
            action = agent.act(state)
            send_action_to_unity(action)

            data = communicate_with_unity()
            if data is None:
                print("等待Unity环境反馈...")
                continue

            # 存储经验
            next_trajectory = data['trajectory']
            next_state = extract_trajectory_features(next_trajectory)
            reward = data['reward']
            done = data['done']
            agent.memory.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # 经验回放
            agent.replay()

        # 更新目标网络
        if e % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

        print(f"Episode {e + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
