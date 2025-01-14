# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


"""
若要添加default，就把required改为Flase，parser.add_argument('--load_model', type=str, default="/home/chd/TJ_Careful/humanoid-gym-main/logs/H1_ppo/exported/policies/policy_1.pt",required=False,
                        help='Run to load from.')

"""

import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import H1_fix_arm_Cfg
import torch

#定义三个类变量，x轴上的线速度，y轴上的线速度，绕z轴的角速度变换率
class cmd:
    vx = 0.4
    vy = 0.0
    dyaw = 0.0

#这个函数将四元数转换为欧拉角数组
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians（弧度）
    return np.array([roll_x, pitch_y, yaw_z])

#这个函数从Mujoco的数据结构data中提取观测信息，用于机器人的状态估计、控制或学习算法中。
def get_obs(data):
    '''Extracts an observation from the mujoco data structure，
    '''
    #获取关节位置，关节速度，获取四元数，使用R.from_quat(quat)从四元数创建一个旋转矩阵，
    #将关节速度dq的前三个元素（通常对应于平移速度）通过旋转矩阵r的逆变换转换到基础框架（base frame）中，
    #从data.sensor('angular-velocity')获取角速度，并转换为双精度浮点数。
    #计算重力向量在基础框架中的表示。
    q = data.qpos.astype(np.double)     #q表示关节位置
    dq = data.qvel.astype(np.double)    #dq表示关节速度
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)  #获取四元数
    r = R.from_quat(quat)   #四元数转变为旋转矩阵
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame，使用了旋转矩阵（通过r表示）来将这些物理量从当前坐标系转换到基础坐标系（世界坐标系）
    omega = data.sensor('imu-angular-velocity').data.astype(np.double)  #角速度的获取
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double) #重力向量的转换，转换到世界坐标
    return (q, dq, quat, v, omega, gvec)

#定义了一个名为pd_control的函数，实现了比例-微分控制算法，用于根据目标位置（target_q）和目标速度（target_dq）来计算所需的控制力矩（或称为“扭矩”）。
def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.
    使用了提供的策略和配置来控制模拟中的实体，

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    #初始化Mujoco模型和数据
    #加载模型
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    #创建数据对象，状态由时间，广义位置和广义速度组成。他们分别是data.time、data.qpos 和 data.qvel。
    #还包含状态的函数，例如物体在世界坐标系中的笛卡尔位置，例如data.geom_xpos
    data = mujoco.MjData(model)
    #表示根据当前的物理状态和模型定义，向前推进一个时间步的模拟。
    mujoco.mj_step(model, data)
    #创建渲染器
    viewer = mujoco_viewer.MujocoViewer(model, data)

    #准备观测历史和环境设置
    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    #双端队列，是一个双向列表，支持从两端快速添加（append）和弹出（pop）元素，而且比普通的列表（list）在两端添加和弹出元素时更高效。
    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0
    #定义机器人的初始角度
    default_angle=np.zeros((cfg.env.num_actions),dtype=np.double)
    default_angle[0]=cfg.init_state.default_joint_angles["left_hip_yaw_joint"]
    default_angle[1]=cfg.init_state.default_joint_angles["left_hip_roll_joint"]
    default_angle[2]=cfg.init_state.default_joint_angles["left_hip_pitch_joint"]
    default_angle[3]=cfg.init_state.default_joint_angles["left_knee_joint"]
    default_angle[4]=cfg.init_state.default_joint_angles["left_ankle_joint"]
    default_angle[5]=cfg.init_state.default_joint_angles["right_hip_yaw_joint"]
    default_angle[6]=cfg.init_state.default_joint_angles["right_hip_roll_joint"]
    default_angle[7]=cfg.init_state.default_joint_angles["right_hip_pitch_joint"]
    default_angle[8]=cfg.init_state.default_joint_angles["right_knee_joint"]
    default_angle[9]=cfg.init_state.default_joint_angles["right_ankle_joint"]
    # default_angle[10]=cfg.init_state.default_joint_angles["torso_joint"]
    # default_angle[11]=cfg.init_state.default_joint_angles["left_shoulder_pitch_joint"]
    # default_angle[12]=cfg.init_state.default_joint_angles["left_shoulder_roll_joint"]
    # default_angle[13]=cfg.init_state.default_joint_angles["left_shoulder_yaw_joint"]
    # default_angle[14]=cfg.init_state.default_joint_angles["left_elbow_joint"]
    # default_angle[15]=cfg.init_state.default_joint_angles["right_shoulder_pitch_joint"]
    # default_angle[16]=cfg.init_state.default_joint_angles["right_shoulder_roll_joint"]
    # default_angle[17]=cfg.init_state.default_joint_angles["right_shoulder_yaw_joint"]
    # default_angle[18]=cfg.init_state.default_joint_angles["right_elbow_joint"]
    # 模拟循环，循环次数由cfg.sim_config.sim_duration / cfg.sim_config.dt决定，
    # cfg.sim_config.sim_duration表示模拟的总时长（可能是以秒为单位），而cfg.sim_config.dt表示模拟的时间步长（每个时间步的持续时间，同样以秒为单位），后被计算出的是模拟的时间总步数。
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation,
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]    #只保留环境中最后cfg.env.num_actions个元素。
        dq = dq[-cfg.env.num_actions:]

        #模拟步骤，模拟步骤低级别模拟步骤的计算器是否是采样率的倍数，如果是则执行以下操作。
        if count_lowlevel % cfg.sim_config.decimation == 0:

            #初始化观测向量
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            #欧拉角计算，将四元数转换为欧拉角，并调整欧拉角，确保它们位于[-pi,pi]区间内
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            #通过某个基于时间的正弦和余弦函数计算得到的
            obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
            obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)

            #线速度，和偏航角速度
            obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel

            #关节角度和关节速度，
            obs[0, 5:15] = (q-default_angle) * cfg.normalization.obs_scales.dof_pos
            obs[0, 15:25] = dq * cfg.normalization.obs_scales.dof_vel
            #上一个控制动作
            obs[0, 25:35] = action

            #
            obs[0, 35:38] = omega
            obs[0, 38:41] = eu_ang

            #观测值裁剪
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            #更新观测历史
            hist_obs.append(obs)
            hist_obs.popleft()

            #准备策略输入
            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            #计算动作
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            #计算目标关节位置
            target_q = action * cfg.control.action_scale+default_angle

        count_lowlevel += 1

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control，生成PD控制
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques

        data.ctrl = tau
        #执行模拟步骤
        mujoco.mj_step(model, data)
        viewer.render()
    #渲染和关闭
    viewer.close()


if __name__ == '__main__':
    import argparse
    #部署一个基于pytorch的模型到Mujoco仿真环境中
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default="/home/chd/TJ_Careful/humanoid-gym/humanoid-gym-main/logs/H1_fix_arm_ppo/exported/policies/policy_1.pt",required=False,
                        help='Run to load from.')
    #--terrain表示是否使用地形
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    #继承的XBotLCfg中包含观察维度，机器人模型设定，地形设定，噪声设定，PD设定，奖励设定等。
    class Sim2simCfg(H1_fix_arm_Cfg):
        #定义mujoco路径
        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/xml/h1_fix_arm.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/xml/h1_fix_arm.xml'
            sim_duration = 60.0
            dt = 0.002
            decimation = 5
        #定义KP，KD
        class robot_config:
            kps = np.array([200, 200, 200,300, 40, 200, 200, 200, 300, 40], dtype=np.double)
            kds = np.array([5, 5, 5, 6, 2, 5, 5, 5, 6, 2], dtype=np.double)
            tau_limit = 200. * np.ones(10, dtype=np.double) #力矩的范围

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
