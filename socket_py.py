import socket
import threading
import numpy as np
import time  # 导入time模块用于实现延迟


# 处理接收 Unity 返回的数据
def receive_data(client_socket, data_event, received_message, stop_event):
    while not stop_event.is_set():  # 在 stop_event 被设置时结束线程
        try:
            data = client_socket.recv(1024).decode('utf-8')
            if not data:
                break
            print(f"Received from Unity: {data}")
            # 更新接收到的数据并通知发送线程
            received_message[0] = data
            data_event.set()  # 设置事件，通知发送线程
        except ConnectionResetError:
            print("Connection lost")
            break
        except OSError:
            # 可能由于关闭 socket 引发的错误
            break


# 启动客户端连接到 Unity
def start_client(host='127.0.0.1', port=9999, num_simulations=5):
    # 定义输入序列
    input_sequence = np.zeros(30)  # 例：输入序列
    input_sequence[5] = -20  # 在序列中间设置一些值

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

    try:
        for sim in range(num_simulations):
            print(f"Starting simulation {sim + 1} of {num_simulations}...\n")
            input_index = 0  # 用于跟踪当前发送位置

            # 发送开始信号
            start_signal = "START"
            client_socket.send(start_signal.encode('utf-8'))
            print(f"Sent to Unity: {start_signal}")
            time.sleep(0.05)  # 等待 50 毫秒

            while input_index <= len(input_sequence):  # 包含结束信号的特殊处理
                if input_index < len(input_sequence):
                    # 如果不是第一次，等待接收到上次的消息
                    if input_index > 0:
                        print("Waiting for Unity's response...")
                        data_event.wait()  # 等待接收到 Unity 的消息
                        print(f"Received message: {received_message[0]}")
                        data_event.clear()  # 重置事件

                    # 获取当前值并发送
                    current_value = input_sequence[input_index]
                    client_socket.send(str(current_value).encode('utf-8'))
                    print(f"Sent to Unity: {current_value}")

                    # 等待 50 毫秒后再发送下一个数据
                    time.sleep(0.08)  # 延迟 50 毫秒

                else:
                    # 发送结束信号
                    end_signal = "END"
                    client_socket.send(end_signal.encode('utf-8'))
                    print(f"Sent to Unity: {end_signal}")

                # 增加索引，准备发送下一个值或结束信号
                input_index += 1

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

    start_client(num_simulations=5)  # 设置仿真次数为 5