# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


#def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    #print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按装订区域中的绿色按钮以运行脚本。
#if __name__ == '__main__':
    #print_hi('PyCharm')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 定义环境和机器人
class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacles = []  # 障碍物列表，每个障碍物是一个矩形区域

    def add_obstacle(self, x, y, width, height):
        self.obstacles.append((x, y, width, height))

    def is_collision(self, x, y, size):
        for obstacle in self.obstacles:
            ox, oy, ow, oh = obstacle
            if (x < ox + ow and x + size > ox and
                y < oy + oh and y + size > oy):
                return True
        return False

class Robot:
    def __init__(self, env, x, y, size):
        self.env = env
        self.x = x
        self.y = y
        self.size = size
        self.path = [(x, y)]
        self.line, = plt.plot([], [], 'bo-')  # 初始化路径线
        self.point, = plt.plot([], [], 'ro')  # 初始化机器人位置点

    def move(self, dx, dy):
        new_x = self.x + dx
        new_y = self.y + dy
        if not self.env.is_collision(new_x, new_y, self.size):
            self.x = new_x
            self.y = new_y
            self.path.append((self.x, self.y))

    def draw(self, ax):
        self.line.set_data([p[0] for p in self.path], [p[1] for p in self.path])
        self.point.set_data(self.x, self.y)
        return self.line, self.point

# 创建环境和机器人
env = Environment(10, 10)
env.add_obstacle(3, 3, 2, 2)
env.add_obstacle(7, 7, 2, 2)

robot = Robot(env, 0.5, 0.5, 0.1)

# 创建绘图
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')

# 动画函数
def animate(frame):
    robot.move(0.1, 0.1)  # 每次移动0.1单位
    return robot.draw(ax)

# 创建动画
ani = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)

plt.show()