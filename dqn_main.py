import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 选择设备：GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
STATE_SIZE = 20  # 假设轨迹特征编码为20维向量
ACTION_SIZE = 2  # 动作定义为 [初始角度变化量, 预弯曲位置变化量]
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10


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
    trajectory = np.array(trajectory)
    if trajectory.shape[0] == 0:
        return np.zeros(STATE_SIZE)

    x = trajectory[:, 0]
    z = trajectory[:, 1]

    # 提取轨迹特征
    features = [
        np.mean(x), np.mean(z),  # 均值
        np.std(x), np.std(z),  # 标准差
        np.min(x), np.min(z),  # 最小值
        np.max(x), np.max(z),  # 最大值
        np.ptp(x), np.ptp(z)  # 极差
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

        self.model = DQN(STATE_SIZE, ACTION_SIZE).to(device)
        self.target_model = DQN(STATE_SIZE, ACTION_SIZE).to(device)
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
        "delta_angle": action[0],
        "delta_bend_position": action[1]
    }
    with open(file_path, "w") as f:
        json.dump(action_dict, f)


# 主训练循环
if __name__ == "__main__":
    agent = DQNAgent()
    num_episodes = 500

    # 记录初始参数
    initial_angle = 0.0
    bend_position = 0.0

    for e in range(num_episodes):
        # 设置机器人参数
        current_params = {"initial_angle": initial_angle, "bend_position": bend_position}
        with open("robot_params.json", "w") as f:
            json.dump(current_params, f)

        print(f"Episode {e + 1}: Initial Angle = {initial_angle}, Bend Position = {bend_position}")

        state_data = communicate_with_unity()
        if state_data is None:
            print("等待Unity环境提供初始状态...")
            continue

        trajectory = state_data['trajectory']
        state = extract_trajectory_features(trajectory)
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            send_action_to_unity(action)

            data = communicate_with_unity()
            if data is None:
                print("等待Unity环境反馈...")
                continue

            next_trajectory = data['trajectory']
            next_state = extract_trajectory_features(next_trajectory)
            reward = data['reward']
            done = data['done']
            agent.memory.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            agent.replay()

        # 更新参数
        initial_angle += action[0]
        bend_position += action[1]

        if e % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

        print(f"Episode {e + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
