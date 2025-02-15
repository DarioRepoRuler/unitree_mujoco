import time
import sys
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.utils.crc import CRC

import torch
import os
from model.actor_critic_new import ActorCritic
from time import sleep
import keyboard
from pynput import keyboard
import threading

def load(path, actor_critic:ActorCritic, device):
        try:
            loaded_dict = torch.load(path, map_location=device, weights_only=True)
            actor_critic.load_state_dict(loaded_dict['model_state_dict'])
            actor_critic.eval()
            print(f"Model loaded successfully from {path}")
        except Exception as e:
                print(f"Error loading the model: {e}")
                raise

# Assuming `model` is already loaded from another script or function
def convert_and_save_model(model, output_model_path):
    """
    Convert the PyTorch model to a TorchScript model and save it.
    """
    try:
        # Use torch.jit.script() for models with dynamic behavior
        scripted_model = torch.jit.script(model)

        # Optionally, you can use torch.jit.trace() if the model has a static computation graph
        # example_input = torch.zeros(1, 48)  # Replace with the actual input shape if known
        # scripted_model = torch.jit.trace(model, example_input)

        # Save the TorchScript model
        scripted_model.save(output_model_path)
        print(f"TorchScript model saved successfully at {output_model_path}")
    except Exception as e:
        print(f"Error converting or saving the model: {e}")
        raise


cfg ={"actor_hidden_dim": 256,"actor_n_layers": 6,"critic_hidden_dim": 256,"critic_n_layers": 6,"std": 1.0}
num_single_obs=48
num_single_priv_obs = 48
device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs = torch.zeros(num_single_obs).to(device)
dq = torch.zeros(12).to(device)

ckpt_path = os.path.join(os.getcwd(),"best_models/rando_all1.pt") #trained_terrain_l2
if not os.path.exists(ckpt_path):
    assert False, f"Model checkpoint not found at {ckpt_path}"

print(f"Loading policy model {ckpt_path}")


actor_critic = ActorCritic(cfg,
                            num_single_obs=num_single_obs,
                            num_obs=num_single_obs,
                            num_priv_obs=num_single_priv_obs,
                            num_actions=12
                            ).to(device)

load(path=ckpt_path, actor_critic=actor_critic, device=device)

#convert_and_save_model(actor_critic, os.path.join(os.getcwd(),"best_models/rando_all15_ts.pt"))
print("Model loaded successfully")


stand_up_joint_pos = np.array([
    -0.1, 0.9, -1.8, -0.1, 0.9, -1.8,
    -0.1, 0.9, -1.8, -0.1, 0.9, -1.8
],
                              dtype=float)

stand_down_joint_pos = np.array([
    0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
    1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
],
                                dtype=float)

default_pos = torch.tensor(
            [0, 0, 0.35, 1, 0, 0, 0, # base coord + quat, former height 0.27
             -0.1, 0.9, -1.8, #FR
             0.1, 0.9, -1.8,  #FL
             -0.1, 0.9, -1.8, #RR
             0.1, 0.9, -1.8]  #RL
        ).to(device)

dt_fast = 0.002
dt_slow = 0.01
runing_time = 0.0
runing_time_slow = 0.0
runing_time_fast = 0.0
crc = CRC()

# I figured out that np arrays are not overwritten by the subscriber callback, the tensors however are infact overwritten!
command = torch.tensor([0.0, 0.0 , 0.0]).to(device)
policy_id = torch.tensor((0,)).to(device)
quaternion = torch.zeros(4)

input("Press enter to start")

def quat_invert(q):
    return q[0], -q[1], -q[2], -q[3]

def quaternion_to_rotation_matrix(q):
    q_w, q_x, q_y, q_z = q
    R = np.array([
        [1 - 2 * (q_y**2 + q_z**2), 2 * (q_x * q_y - q_w * q_z), 2 * (q_x * q_z + q_w * q_y)],
        [2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_x**2 + q_z**2), 2 * (q_y * q_z - q_w * q_x)],
        [2 * (q_x * q_z - q_w * q_y), 2 * (q_y * q_z + q_w * q_x), 1 - 2 * (q_x**2 + q_y**2)]
    ])
    return R

def rotate_vector(q, g_world):
        R = quaternion_to_rotation_matrix(q)
        g_body = R @ g_world
        return g_body

def get_gravity_vector(quaternion):
    g = np.array([0, 0, -1])
    quaternion = quat_invert(quaternion)
    proj_grav = rotate_vector(quaternion, g)
    return proj_grav

def LowStateHandler(msg: LowState_):
    
    
    #print("Accelerometer: ", msg.imu_state.accelerometer)
    # accel = torch.tensor(msg.imu_state.accelerometer).to(device)
    # print(f"Accelerometer: {accel+torch.tensor([-0.1838, +0.0188, -9.81]).to(device)}")
    #proj_gravity = proj_gravity / torch.norm(proj_gravity)

    quaternion[:] = torch.tensor(msg.imu_state.quaternion)
    ang_vel = np.array(msg.imu_state.gyroscope)
    local_w = rotate_vector(quat_invert(np.array(quaternion)), ang_vel)
    obs[3:6]=torch.tensor(ang_vel).to(device) * 0.25
    proj_gravity = get_gravity_vector(np.array(quaternion))
    obs[6:9]=torch.tensor(proj_gravity).to(device)
    
    q=torch.zeros(12).to(device)
    qvel=torch.zeros(12).to(device)
    for i in range(12):
        q[i]=msg.motor_state[i].q
        qvel[i]=msg.motor_state[i].dq
    dq =qvel

    obs[9:21]=q-default_pos[7:]
    obs[21:33]=qvel*0.1


def HighStateHandler(msg: SportModeState_):
    glob__lin_vel = np.array(msg.velocity)
    local_vel = rotate_vector(quat_invert(quaternion), glob__lin_vel)*2.0
    obs[:3]=torch.tensor(local_vel).to(device)

def on_press(key):
    global stop_loop
    try:
        # Check if the key is one of the arrow keys
        if key == keyboard.Key.up:
            command[0] = torch.clamp(command[0]+0.1, -1.2, 1.2)
            #command[2] = 0.0
            print(f"Command: {command}")
        elif key == keyboard.Key.down:
            command[0] = torch.clamp(command[0]-0.1, -1.2, 1.2)
            #command[2] = 0.0
            print(f"Command: {command}")
        elif key == keyboard.Key.left:
            #command[0] = 0.0
            command[2] = torch.clamp(command[2]+0.1, -1.2, 1.2)
            print(f"Command: {command}")
        elif key == keyboard.Key.right:
            #command[0] = 0.0
            command[2] = torch.clamp(command[2]-0.1, -1.2, 1.2)
            print(f"Command: {command}")
        elif key.char == 'e':
            command[:] = 0.0
            policy_id[0] = 1
            print(f"Commanded policy: {policy_id}")
        
        elif key.char == 'd':
            print(f"Commanded laying down")
            command[:] = 0.0
            policy_id[0] = 2
        
        elif key.char == 's':
            policy_id[0] = 0
        elif key.char == '0':
            command[:]=0.0

        
        # Check if the 'q' key is pressed to stop the loop
        elif key.char == 'q':
            stop_loop = True
            return False  # Stop listener

    except AttributeError:
        # Handle special keys
        print(f"Special key pressed: {key}")

        

def start_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == '__main__':
    pol_id = 0
    if len(sys.argv) <2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])
    
    last_action = torch.zeros(12).to(device)
    obs[45:]=command*2.0

    # Create a publisher to publish the data defined in UserData class
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    low_state_suber = ChannelSubscriber("rt/lowstate", LowState_)
    low_state_suber.Init(LowStateHandler, 10)

    hight_state_suber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    hight_state_suber.Init(HighStateHandler, 10)
    low_state_suber.Init(LowStateHandler, 10)

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0

    timeoffset = None
    lower_limits = torch.tensor([-1.0472, -1.5708, -2.7227]*2 + [-1.0472, -0.5236, -2.7227]*2).to(device)
    upper_limits = torch.tensor([1.0472, 3.4907, -0.83776]*2 + [1.0472, 4.5379, -0.83776]*2).to(device)
    m = (lower_limits + upper_limits) / 2
    r = upper_limits - lower_limits
    lower_limits = m - 0.5 * r * 0.9
    upper_limits = m + 0.5 * r * 0.9

    ctrl_cnt=-1

    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    # Start the listener in a separate thread
    listener_thread = threading.Thread(target=start_listener)
    listener_thread.start()

    while True:
        obs[45:]=command *2.0
        obs[33:45]=last_action

        step_start_slow = time.perf_counter()

        
        
        runing_time_slow += dt_slow
        runing_time_fast =0.0
        total_fast_time = 0.0

        if policy_id[0]==0:
            actions = actor_critic.forward(obs)
            target_dof_pos = actions*0.3 + default_pos[7:]
            last_action = actions
        
        while total_fast_time < dt_slow:


            step_start_fast = time.perf_counter()

            runing_time_fast += dt_fast
            
            if (runing_time_slow < 3.0) and policy_id[0]==0:
                # Stand up in first 3 ssecond
                print(f"Stand up")
                # Total time for standing up or standing down is about 1.2s
                phase = np.tanh(runing_time_slow / 1.2)
                for i in range(12):
                    cmd.motor_cmd[i].q = phase * stand_up_joint_pos[i] + (
                        1 - phase) * stand_down_joint_pos[i]

                    cmd.motor_cmd[i].kp = phase * 55.0 + (1 - phase) * 20.0
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = 1.0
                    cmd.motor_cmd[i].tau = 0.0

            # Execute policy
            elif (runing_time_slow > 3.0) and policy_id[0]==0:
                for i in range(12):
                    cmd.motor_cmd[i].q = target_dof_pos[i] #phase * stand_down_joint_pos[i] + (1 - phase) * stand_up_joint_pos[i] #
                    cmd.motor_cmd[i].kp = 55.0
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = 1.
                    cmd.motor_cmd[i].tau = 0.0
            
            elif policy_id[0]==1:
                # Enter damping mode
                print(f"[E]mergency: in damping mode")
                if timeoffset is None:
                    timeoffset = runing_time_slow
                for i in range(12):
                    cmd.motor_cmd[i].q = 0.0
                    cmd.motor_cmd[i].kp = 0.0
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd =1.0
                    cmd.motor_cmd[i].tau = 0.0
                if runing_time_slow > timeoffset + 2.0:
                    exit(1)

                

            elif policy_id==2:
                command[:] = 0.0
                print(f"[D]own ")
                if timeoffset is None:
                    timeoffset = runing_time_slow
                # Then stand down
                phase = np.tanh((runing_time_slow - timeoffset) / 1.2)
                
                for i in range(12):
                    cmd.motor_cmd[i].q = phase * stand_down_joint_pos[i] + (
                        1 - phase) * stand_up_joint_pos[i]
                    cmd.motor_cmd[i].kp = 30.0
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = 1.0
                    cmd.motor_cmd[i].tau = 0.0

                if runing_time_slow > timeoffset + 2.0:
                    runing_time_slow = 0.0
                    policy_id[0]=-1
                    timeoffset = None


            
            elif policy_id[0]==-1:
                print(f"StandBy")
                runing_time_slow = 0.0
                cmd.motor_cmd[i].q = 0.0
                cmd.motor_cmd[i].kp = 0.0
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd =3.5
                cmd.motor_cmd[i].tau = 0.0
                               
            
            cmd.crc = crc.Crc(cmd)
            pub.Write(cmd)
            
            total_fast_time = time.perf_counter() - step_start_slow
            #print(f"Total fast time: {total_fast_time}")
            time_until_next_step = dt_fast - (time.perf_counter() - step_start_fast)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        #print(f"Total slow time: {time.perf_counter() - step_start_slow}")
        time_until_next_step = dt_slow - (time.perf_counter() - step_start_slow)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
