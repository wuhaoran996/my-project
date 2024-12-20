import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math

# Q网络定义
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(10000)
        self.batch_size = 64

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # 随机选择动作（探索）
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return torch.argmax(q_values, dim=1).item()  # 选择最大Q值对应的动作（利用）

    def update(self):
        if self.buffer.size() < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.max(1)[0].unsqueeze(1)
            target_q_value = rewards + (self.gamma * next_q_value * (1 - dones))

        # 当前Q值
        q_values = self.q_network(states).gather(1, actions)

        # 计算损失
        loss = nn.MSELoss()(q_values, target_q_value)

        # 反向传播并更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def apply_action_constraints(self, action):
        """
        应用动作限制规则：
        - 如果上一步是弯曲，则后续五步必须直行。
        """
        if self.last_action_was_turn and self.steps_after_turn < 5:
            # 后五步必须直行
            if action != 0:  # 如果是弯曲
                return 0  # 强制选择直行
            self.steps_after_turn += 1
        else:
            self.last_action_was_turn = False
            self.steps_after_turn = 0

        # 如果是弯曲，标记为弯曲
        if action == 1 or action == 2:  # 假设1是向左弯曲，2是向右弯曲
            self.last_action_was_turn = True
            self.steps_after_turn = 0  # 重置步数

        return action

def calculate_reward(current_position, target_position, path_length):
    """
    计算奖励：路径越短，奖励越高；距离目标越远，奖励越高。
    """
    # 计算距离目标点的距离
    distance_to_target = math.sqrt((current_position[0] - target_position[0])**2 +
                                   (current_position[1] - target_position[1])**2)
    reward = -path_length + distance_to_target  # 路径越短奖励越高，离目标越远奖励越高
    return reward


# 训练循环
num_episodes = 1000
agent = DQNAgent(state_size=10, action_size=12)  # 例如状态维度为10，动作数为12

for episode in range(num_episodes):
    state, target_position, obstacles = load_robot_state()  # 获取当前状态和目标
    done = False
    total_reward = 0
    path_length = 0  # 路径长度

    while not done:
        action = agent.act(state)  # 选择一个动作
        action = agent.apply_action_constraints(action)  # 应用动作限制

        # 执行动作并得到新的状态（Unity仿真）
        new_state, reward, done = step_in_unity(action)

        # 计算奖励
        reward = calculate_reward(new_state[0], target_position, path_length)

        # 更新路径长度
        path_length += 1

        agent.buffer.push((state, action, reward, new_state, done))  # 将经验加入缓冲区
        agent.update()  # 更新Q网络
        state = new_state  # 更新状态
        total_reward += reward  # 累加奖励

    # 每100个episode更新目标网络
    if episode % 100 == 0:
        agent.update_target_network()

    print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}")
