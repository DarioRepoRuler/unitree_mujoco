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
from model.actor_critic_new_class import ActorCritic
from time import sleep
import keyboard
from pynput import keyboard
import threading
from omegaconf import DictConfig, OmegaConf
import hydra

import yaml


class PolicyVicClass:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.stiff_range = cfg.env.control.stiff_range
        self.p_gains = torch.ones(12) * cfg.env.control.p_gain
        self.actor_critic = None
        self.num_single_obs = cfg.env.single_obs_size
        self.num_single_priv_obs = cfg.env.single_obs_size_priv
        self.num_actions = 12
        self.control_mode = cfg.env.control_mode
        self.convert_to_torch_script = cfg.convert_to_torch_script
        if self.control_mode == "P":
            self.num_actions += 0
            self.num_single_obs += 0
            self.num_single_priv_obs += 0
        elif self.control_mode == "VIC_1":
            self.num_actions += 3
            self.num_single_obs += 3
            self.num_single_priv_obs += 3
        elif self.control_mode == "VIC_2":
            self.num_actions += 4
            self.num_single_obs += 4
            self.num_single_priv_obs += 4
        elif self.control_mode == "VIC_3":
            self.num_actions += 12
            self.num_single_obs += 12
            self.num_single_priv_obs += 12
        elif self.control_mode == "VIC_4":
            self.num_actions += 7
            self.num_single_obs += 7
            self.num_single_priv_obs += 7
        else:
            assert False, f"Control mode {self.control_mode} not supported"
        self.obs = torch.zeros(self.num_single_obs).to(self.device)
        self.last_vel = torch.tensor([0.0,0.0,0.0]).to(self.device)
        self.policy_cfg = cfg.policy
        print(f"Policy config: {self.policy_cfg}")
        self.ckpt_path = os.path.join(os.getcwd(), cfg.ckpt_path)
        print(f"Ckpt path: {self.ckpt_path}")
        self.load_model()

        

        self.stand_down_joint_pos = np.array([
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, -0.15,
            1.22187, -2.44375, 0.15, 1.22187, -2.44375
        ], dtype=float)

        self.default_pos = torch.tensor(self.cfg.env.default_pos).to(self.device)
        self.stand_up_joint_pos = np.array(self.cfg.env.default_pos, dtype=float)[7:]
        self.action_scale = self.cfg.env.action_scale
        self.p_gain = self.cfg.env.control.p_gain
        self.d_gain = self.cfg.env.control.d_gain   

        self.dt_fast = 0.001
        self.dt_slow = 0.02
        self.runing_time = 0.0
        self.runing_time_slow = 0.0
        self.runing_time_fast = 0.0
        self.crc = CRC()

        self.command = torch.tensor([0.0, 0.0, 0.0]).to(self.device)
        self.policy_id = torch.tensor((0,)).to(self.device)
        self.quaternion = torch.zeros(4)

    def load_model(self):

        if not os.path.exists(self.ckpt_path):
            assert False, f"Model checkpoint not found at {self.ckpt_path}"

        print(f"Loading policy model {self.ckpt_path}")

        actor_critic = ActorCritic(self.policy_cfg,
                                   num_single_obs=self.num_single_obs,
                                   num_obs=self.num_single_obs,
                                   num_priv_obs=self.num_single_priv_obs,
                                   num_actions=self.num_actions
                                   ).to(self.device)

        self.load(path=self.ckpt_path, actor_critic=actor_critic, device=self.device)

        self.actor_critic = actor_critic
        print("Model loaded successfully")
        print(f"Output name: {self.cfg.env.control_mode}")
        if self.convert_to_torch_script:
            print(f"Converting model to TorchScript")
            if not os.path.exists(os.path.join(os.getcwd(), 'torchscript_model')):
                os.makedirs(os.path.join(os.getcwd(), 'torchscript_model'))
            output_model_path = os.path.join(os.getcwd(), 'torchscript_model',f"{self.cfg.env.control_mode}_model.pt")
            self.convert_and_save_model(self.actor_critic, output_model_path)
            print(f"Model converted to torchscript and saved to {output_model_path}")


    def load(self, path, actor_critic, device):
        try:
            loaded_dict = torch.load(path, map_location=device, weights_only=False)
            actor_critic.load_state_dict(loaded_dict['model_state_dict'])
            actor_critic.eval()
            print(f"Model loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading the model: {e}")
            raise

    def convert_and_save_model(self, model, output_model_path):
        try:
            scripted_model = torch.jit.script(model)
            scripted_model.save(output_model_path)
            print(f"TorchScript model saved successfully at {output_model_path}")
        except Exception as e:
            print(f"Error converting or saving the model: {e}")
            raise

    def quat_invert(self, q):
        return q[0], -q[1], -q[2], -q[3]

    def quaternion_to_rotation_matrix(self, q):
        q_w, q_x, q_y, q_z = q
        R = np.array([
            [1 - 2 * (q_y ** 2 + q_z ** 2), 2 * (q_x * q_y - q_w * q_z), 2 * (q_x * q_z + q_w * q_y)],
            [2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_x ** 2 + q_z ** 2), 2 * (q_y * q_z - q_w * q_x)],
            [2 * (q_x * q_z - q_w * q_y), 2 * (q_y * q_z + q_w * q_x), 1 - 2 * (q_x ** 2 + q_y ** 2)]
        ])
        return R

    def rotate_vector(self, q, g_world):
        R = self.quaternion_to_rotation_matrix(q)
        g_body = R @ g_world
        return g_body

    def get_gravity_vector(self, quaternion):
        g = np.array([0, 0, -1])
        quaternion = self.quat_invert(quaternion)
        proj_grav = self.rotate_vector(quaternion, g)
        return proj_grav

    def LowStateHandler(self, msg: LowState_):
        self.quaternion[:] = torch.tensor(msg.imu_state.quaternion)
        ang_vel = np.array(msg.imu_state.gyroscope)
        local_w = self.rotate_vector(self.quat_invert(np.array(self.quaternion)), ang_vel)
        self.obs[3:6] = torch.tensor(ang_vel).to(self.device) * 0.25
        proj_gravity = self.get_gravity_vector(np.array(self.quaternion))
        self.obs[6:9] = torch.tensor(proj_gravity).to(self.device)

        q = torch.zeros(12).to(self.device)
        qvel = torch.zeros(12).to(self.device)
        for i in range(12):
            q[i] = msg.motor_state[i].q
            qvel[i] = msg.motor_state[i].dq
        dq = qvel

        self.obs[9:21] = q - self.default_pos[7:]
        self.obs[21:33] = qvel * 0.1

    def HighStateHandler(self, msg: SportModeState_):
        glob__lin_vel = np.array(msg.velocity)
        local_vel = self.rotate_vector(self.quat_invert(self.quaternion), glob__lin_vel) * 2.0
        # self.obs[:3] =torch.tensor(local_vel).to(self.device)
        self.obs[:3] = torch.tensor([0.0,0.0,0.0]).to(self.device)

    def on_press(self, key):
        global stop_loop
        try:
            if key == keyboard.Key.up or key.char.lower() == 'w':
                self.command[0] = torch.clamp(self.command[0] + 0.1, -1.2, 1.2)
            elif key == keyboard.Key.down or key.char.lower() == 's':
                self.command[0] = torch.clamp(self.command[0] - 0.1, -1.2, 1.2)
            elif key == keyboard.Key.left or key.char =='a':
                self.command[1] = torch.clamp(self.command[1] + 0.1, -1.2, 1.2)
            elif key == keyboard.Key.right or key.char == 'd':
                self.command[1] = torch.clamp(self.command[1] - 0.1, -1.2, 1.2)
            elif key.char == 'A':
                self.command[2] = torch.clamp(self.command[2] + 0.1, -1.2, 1.2)
            elif key.char == 'D':
                self.command[2] = torch.clamp(self.command[2] - 0.1, -1.2, 1.2)
            elif key.char == 'e':
                self.command[:] = 0.0
                self.policy_id[0] = 1
            elif key.char == 'l':
                print(f"[L]aying down")
                self.command[:] = 0.0
                self.policy_id[0] = 2
            elif key == keyboard.Key.space or key.char == ' ':  # Changed from 'x' to space key
                print(f"Stand mode activated")
                self.policy_id[0] = 0
            elif key.char == 'n':
                self.command[:] = 0.0
            elif key.char == 'q':
                stop_loop = True
                return False
        except AttributeError:
            # This handles special keys that don't have a char attribute
            if key == keyboard.Key.space:  # Additional check for space in the AttributeError handler
                print(f"Stand mode activated")
                self.policy_id[0] = 0
            else:
                print(f"Special key pressed: {key}")

    def start_listener(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()
        
    def unscale(self, x, min, max):
        m = (min + max) / 2
        r = (max - min) / 2
        return m + x * r
    
    def test(self, config: DictConfig):
        print(f"Configuration: {config}")
        pol_id = 0
        if len(sys.argv) < 2:
            ChannelFactoryInitialize(1, "lo")
        else:
            ChannelFactoryInitialize(0, sys.argv[1])

        last_action = torch.zeros(self.num_actions).to(self.device)
        self.obs[-3:] = self.command * 2.0

        pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        pub.Init()

        low_state_suber = ChannelSubscriber("rt/lowstate", LowState_)
        low_state_suber.Init(self.LowStateHandler, 10)

        hight_state_suber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        hight_state_suber.Init(self.HighStateHandler, 10)
        low_state_suber.Init(self.LowStateHandler, 10)

        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0

        timeoffset = None
        lower_limits = torch.tensor([-1.0472, -1.5708, -2.7227] * 2 + [-1.0472, -0.5236, -2.7227] * 2).to(self.device)
        upper_limits = torch.tensor([1.0472, 3.4907, -0.83776] * 2 + [1.0472, 4.5379, -0.83776] * 2).to(self.device)
        m = (lower_limits + upper_limits) / 2
        r = upper_limits - lower_limits
        lower_limits = m - 0.5 * r * 0.9
        upper_limits = m + 0.5 * r * 0.9

        ctrl_cnt = -1

        for i in range(20):
            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 0.0

        listener_thread = threading.Thread(target=self.start_listener)
        listener_thread.start()

        while True:
            self.obs[-3:] = self.command * 2.0
            self.obs[33:-3] = last_action

            step_start_slow = time.perf_counter()

            self.runing_time_slow += self.dt_slow
            self.runing_time_fast = 0.0
            total_fast_time = 0.0

            if self.policy_id[0] == 0:
                m = (self.stiff_range[0] + self.stiff_range[1]) / 2
                r = (self.stiff_range[1] - self.stiff_range[0]) / 2
                # print(f"Observations : {self.obs}")
                actions = self.actor_critic.forward(self.obs)
                last_action = actions
                scaled_action = actions[:12] * self.action_scale
                # print(f"Actions: {scaled_action}")
                scaled_action[0] = scaled_action[0] * self.cfg.env.hip_scale
                scaled_action[3] = scaled_action[3] * self.cfg.env.hip_scale
                scaled_action[6] = scaled_action[6] * self.cfg.env.hip_scale
                scaled_action[9] = scaled_action[9] * self.cfg.env.hip_scale
                target_dof_pos = scaled_action + self.default_pos[7:]
                # print(f"target_dof_pos: {target_dof_pos}")
                Kp= torch.ones(12)
                if self.control_mode == "VIC_1":
                    action_stiffness = torch.tile(actions[12:], (4,))
                    Kp = self.p_gains * (m + r * action_stiffness)
                if self.control_mode == "VIC_2":
                    action_stiffness = torch.tile(actions[12:], (3,))
                    Kp = self.p_gains * (m + r * action_stiffness)
                if self.control_mode== "VIC_3":
                    Kp = self.p_gains * (m + r * actions[12:])
                if self.control_mode == "VIC_4":
                    stiff_leg = torch.tile(self.unscale(actions[12:12+4], 0., 1.), (3,)).reshape(3,4)
                    stiff_joint = self.unscale(actions[12+4:12+7], 0., 1.)
                    # Step 3: Compute action_stiff
                    action_stiff = torch.flatten((stiff_leg * stiff_joint.unsqueeze(1)).T)
                    # Step 4: Rescale action_stiff
                    action_stiff = self.unscale(2.0 * (action_stiff - 0.5), self.stiff_range[0], self.stiff_range[1])
                    Kp = self.p_gains * action_stiff
                Kd= 0.2*torch.sqrt(Kp)

            while total_fast_time < self.dt_slow:
                step_start_fast = time.perf_counter()
                self.runing_time_fast += self.dt_fast

                if self.runing_time_slow < 3.0 and self.policy_id[0] == 0:
                    print(f"Stand up")
                    phase = np.tanh(self.runing_time_slow / 1.2)
                    for i in range(12):
                        cmd.motor_cmd[i].q = phase * self.stand_up_joint_pos[i] + (
                                1 - phase) * self.stand_down_joint_pos[i]

                        cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0
                        cmd.motor_cmd[i].dq = 0.0
                        cmd.motor_cmd[i].kd = 1.0
                        cmd.motor_cmd[i].tau = 0.0

                elif self.runing_time_slow > 3.0 and self.policy_id[0] == 0:
                    print(f"Running")
                    if "VIC" in self.control_mode:  
                        for i in range(12):
                            cmd.motor_cmd[i].q = target_dof_pos[i]
                            cmd.motor_cmd[i].kp = Kp[i]
                            cmd.motor_cmd[i].dq = 0.0
                            cmd.motor_cmd[i].kd = Kd[i]
                            cmd.motor_cmd[i].tau = 0.0
                    if self.control_mode == "P":
                        for i in range(12):
                            cmd.motor_cmd[i].q = target_dof_pos[i]
                            cmd.motor_cmd[i].kp = self.p_gain
                            cmd.motor_cmd[i].dq = 0.0
                            cmd.motor_cmd[i].kd = self.d_gain
                            cmd.motor_cmd[i].tau = 0.0

                elif self.policy_id[0] == 1:
                    print(f"[E]mergency: in damping mode")
                    if timeoffset is None:
                        timeoffset = self.runing_time_slow
                    for i in range(12):
                        cmd.motor_cmd[i].q = 0.0
                        cmd.motor_cmd[i].kp = 0.0
                        cmd.motor_cmd[i].dq = 0.0
                        cmd.motor_cmd[i].kd = 1.0
                        cmd.motor_cmd[i].tau = 0.0
                    if self.runing_time_slow > timeoffset + 2.0:
                        exit(1)

                elif self.policy_id == 2:
                    self.command[:] = 0.0
                    print(f"[D]own")
                    if timeoffset is None:
                        timeoffset = self.runing_time_slow
                    phase = np.tanh((self.runing_time_slow - timeoffset) / 1.2)

                    for i in range(12):
                        cmd.motor_cmd[i].q = phase * self.stand_down_joint_pos[i] + (
                                1 - phase) * self.stand_up_joint_pos[i]
                        cmd.motor_cmd[i].kp = 30.0
                        cmd.motor_cmd[i].dq = 0.0
                        cmd.motor_cmd[i].kd = 1.0
                        cmd.motor_cmd[i].tau = 0.0

                    if self.runing_time_slow > timeoffset + 2.0:
                        self.runing_time_slow = 0.0
                        self.policy_id[0] = -1
                        timeoffset = None

                elif self.policy_id[0] == -1:
                    print(f"StandBy")
                    self.runing_time_slow = 0.0
                    for i in range(12):
                        cmd.motor_cmd[i].q = 0.0
                        cmd.motor_cmd[i].kp = 0.0
                        cmd.motor_cmd[i].dq = 0.0
                        cmd.motor_cmd[i].kd = 3.5
                        cmd.motor_cmd[i].tau = 0.0

                cmd.crc = self.crc.Crc(cmd)
                pub.Write(cmd)

                total_fast_time = time.perf_counter() - step_start_slow
                time_until_next_step = self.dt_fast - (time.perf_counter() - step_start_fast)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            time_until_next_step = self.dt_slow - (time.perf_counter() - step_start_slow)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

@hydra.main(config_path='config', config_name='test_pos_hw', version_base="1.2")
def test(cfg: DictConfig):
    print(f"Configuration: {cfg}")
    OmegaConf.set_struct(cfg, False)
    cfg.convert_to_torch_script = True
    policy_vic = PolicyVicClass(cfg)
    policy_vic.test(cfg)


if __name__ == '__main__':
    test()

    


