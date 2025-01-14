import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, Lock
from collections import deque

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

# Constants
dt = 0.002
runing_time = 0.0
crc = CRC()
MAX_DATA_POINTS = 500  # Limit the size of data for plotting

# Data structures for live plotting with thread-safe operations
kp_values = {f'kp_{i}': deque(maxlen=MAX_DATA_POINTS) for i in range(12)}
torque_values = {f'torque_{i}': deque(maxlen=MAX_DATA_POINTS) for i in range(12)}
data_lock = Lock()  # To protect shared data between threads

# Define a function to update the live plot for kp
def live_plot_kp():
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Live kp Values")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Stiffness (Kp)")
    lines = [ax.plot([], [], label=f'kp_{i}')[0] for i in range(12)]
    ax.legend()
    update_interval = 0.1  # Time in seconds between plot updates

    while True:
        time.sleep(update_interval)
        with data_lock:
            for i, line in enumerate(lines):
                line.set_data(range(len(kp_values[f'kp_{i}'])), list(kp_values[f'kp_{i}']))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

# Define a function to update the live plot for torque
def live_plot_torque():
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Live torque Values")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Torque [Nm]")
    lines = [ax.plot([], [], label=f'torque_{i}')[0] for i in range(12)]
    ax.legend()
    update_interval = 0.1  # Time in seconds between plot updates

    while True:
        time.sleep(update_interval)
        with data_lock:
            for i, line in enumerate(lines):
                line.set_data(range(len(torque_values[f'torque_{i}'])), list(torque_values[f'torque_{i}']))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

def LowStateHandler(msg: LowState_):
    with data_lock:
        for i in range(12):
            torque_values[f'torque_{i}'].append(msg.motor_state[i].tau_est)

def LowCmdHandler(msg: LowCmd_):
    with data_lock:
        for i in range(12):
            kp_values[f'kp_{i}'].append(msg.motor_cmd[i].kp)

input("Press enter to start")

if __name__ == '__main__':
    # Start the live plotting threads
    Thread(target=live_plot_kp, daemon=True).start()
    Thread(target=live_plot_torque, daemon=True).start()

    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
    low_cmd_suber.Init(LowCmdHandler, 10)
    low_state_suber = ChannelSubscriber("rt/lowstate", LowState_)
    low_state_suber.Init(LowStateHandler, 10)

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    while True:
        step_start = time.perf_counter()
        runing_time += dt
        cmd.crc = crc.Crc(cmd)
        time.sleep(max(0, dt - (time.perf_counter() - step_start)))
