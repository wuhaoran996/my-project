import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数配置
STATE_SIZE = 20      # 假设轨迹特征为20维
ACTION_SIZE = 2      # 动作空间: 初始角度变化量与预弯曲位置变化量
GAMMA = 0.99         # 奖励折扣因子
LR_ACTOR = 0.001     # Actor学习率
LR_CRITIC = 0.002    # Critic学习率
TAU = 0.005          # 目标网络软更新速率
BATCH_SIZE = 64      # 每批训练样本大小
MEMORY_SIZE = 10000  # 经验池大小
EPSILON = 1.0        # 探索因子
EPSILON_MIN = 0.01   # 最小探索因子
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10  # 更新目标网络的频率

# 经验回放池
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

# 状态归一化
def normalize_state(state, state_min=-10, state_max=10):
    return np.clip((state - state_min) / (state_max - state_min), -1, 1)

# Actor 网络：输出连续动作
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # 输出连续动作，归一化到[-1, 1]

# Critic 网络：评估状态-动作对的值函数
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(action_size, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        y = torch.relu(self.fc2(action))
        z = torch.relu(x + y)
        return self.fc3(z)

# DDPG Agent
class DDPGAgent:
    def __init__(self):
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.memory = ReplayBuffer(MEMORY_SIZE)

        # 初始化Actor和Critic网络
        self.actor = Actor(STATE_SIZE, ACTION_SIZE).to(device)
        self.target_actor = Actor(STATE_SIZE, ACTION_SIZE).to(device)
        self.critic = Critic(STATE_SIZE, ACTION_SIZE).to(device)
        self.target_critic = Critic(STATE_SIZE, ACTION_SIZE).to(device)

        # 更新目标网络
        self.update_target_networks()

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # 初始epsilon
        self.epsilon = EPSILON

    def update_target_networks(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state_tensor)
        return action.cpu().numpy()[0]

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # 从经验池中随机采样一批数据
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # 更新Critic网络
        next_actions = self.target_actor(next_states)
        target_q_values = rewards + (1 - dones) * GAMMA * self.target_critic(next_states, next_actions).detach()

        current_q_values = self.critic(states, actions)

        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor网络
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()  # 最大化Q值
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

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

# 状态提取函数（假设从轨迹中提取特征）
def extract_trajectory_features(trajectory):
    # 假设轨迹为一系列散点，提取合适的特征（如坐标、速度等）
    return np.mean(trajectory, axis=0)  # 这里只是一个示例，具体要根据任务设计

# 主训练循环
if __name__ == "__main__":
    agent = DDPGAgent()
    num_episodes = 500

    initial_angle = 0.0
    bend_position = 0.0

    for e in range(num_episodes):
        # 设置机器人初始参数
        current_params = {"initial_angle": initial_angle, "bend_position": bend_position}
        with open("robot_params.json", "w") as f:
            json.dump(current_params, f)

        print(f"Episode {e + 1}: Initial Angle = {initial_angle}, Bend Position = {bend_position}")

        # 获取Unity环境的状态数据
        state_data = communicate_with_unity()
        if state_data is None:
            print("等待Unity环境提供初始状态...")
            continue

        trajectory = state_data['trajectory']
        state = np.array([normalize_state(feature) for feature in extract_trajectory_features(trajectory)])  # 归一化状态
        total_reward = 0
        done = False

        while not done:
            # Actor选择动作
            action = agent.act(state)
            send_action_to_unity(action)

            data = communicate_with_unity()
            if data is None:
                print("等待Unity环境反馈...")
                continue

            next_trajectory = data['trajectory']
            next_state = np.array([normalize_state(feature) for feature in extract_trajectory_features(next_trajectory)])  # 归一化
            reward = data['reward']
            done = data['done']
            agent.memory.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            agent.replay()

        # 更新机器人参数
        initial_angle += action[0]
        bend_position += action[1]

        # 每训练一定次数更新目标网络
        if e % TARGET_UPDATE_FREQ == 0:
            agent.update_target_networks()

        print(f"Episode {e + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        # 最终的训练完成输出
        print("训练完成！")

        # 保存训练好的模型
        torch.save(agent.actor.state_dict(), "actor_model.pth")
        torch.save(agent.critic.state_dict(), "critic_model.pth")
        print("模型已保存！")

