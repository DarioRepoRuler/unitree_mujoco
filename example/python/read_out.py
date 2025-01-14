import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


dt = 0.002
runing_time = 0.0
crc = CRC()

# Initialize a list to store kp values for live plotting
kp_values = {f'kp_{i}': [] for i in range(12)}

# Define a function to update the live plot
def live_plot():
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Live kp Values")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Stiffness (Kp)")
    lines = []
    for i in range(12):
        line, = ax.plot([], [], label=f'kp_{i}')
        lines.append(line)
    ax.legend()

    while True:
        for i, line in enumerate(lines):
            line.set_ydata(kp_values[f'kp_{i}'])
            line.set_xdata(range(len(kp_values[f'kp_{i}'])))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)  # Adjust sleep to balance CPU usage and responsiveness


def LowStateHandler(msg: LowState_):
    foot_force = msg.foot_force
    print("foot_force: ", foot_force)
    
def LowCmdHandler(msg: LowCmd_):
    print("LowCmdHandler")
    print(f"Motor command: {msg.motor_cmd[0].kp}")
    for i in range(12):
        kp_values[f'kp_{i}'].append(msg.motor_cmd[i].kp)
        # Limit the size of kp_values to avoid memory issues
        if len(kp_values[f'kp_{i}']) > 1000:
            kp_values[f'kp_{i}'].pop(0)

input("Press enter to start")

if __name__ == '__main__':
    # Start the live plotting thread
    plot_thread = Thread(target=live_plot, daemon=True)
    plot_thread.start()

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

        # time_until_next_step = dt - (time.perf_counter() - step_start)
        # if time_until_next_step > 0:
        time.sleep(0.01)
