import argparse
import time
import torch
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from Planner import OkayPlan
from Env import DynamicEnv, str2bool

'''Configurations:'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='Running device of SEPSO_Down, cuda or cpu')
parser.add_argument('--RO', type=str2bool, default=True, help='Random_Obs; True = random obstacles; False = consistent obstacles')
parser.add_argument('--DPI', type=str2bool, default=True, help='True for DPI(from OkayPlan), False for PI(from SEPSO)')
parser.add_argument('--KP', type=str2bool, default=True, help='whether to use Kinematics_Penalty')
parser.add_argument('--FPS', type=int, default=0, help='Render FPS, 0 for maximum speed')
parser.add_argument('--Playmode', type=str2bool, default=True, help='Play with keyboard: UP, DOWN, LEFT, RIGHT')

# Planner related:
parser.add_argument('--Max_iterations', type=int, default=50, help='maximum number of particle iterations for each planning')
parser.add_argument('--N', type=int, default=170, help='number of particles in each group')
parser.add_argument('--D', type=int, default=14, help='particle dimension: number of waypoints = D/2')
parser.add_argument('--Quality', type=float, default=10, help='planning quality: the smaller, the better quality, and the longer time')

# Env related:
parser.add_argument('--window_size', type=int, default=366, help='render window size, minimal: 366')
parser.add_argument('--seg', type=int, default=4, help='number of segments of the obstacle: 4/6/10')
parser.add_argument('--Obstacle_V', type=int, default=4, help='maximal x/y velocity of the obstacles')
parser.add_argument('--Target_V', type=int, default=3, help='velocity of the target')
parser.add_argument('--Start_V', type=int, default=6, help='velocity of the robot')
parser.add_argument('--Record_freq', type=int, default=-1, help='Record frequency; Negative=No recording')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)
opt.Search_range = (0., opt.window_size)
if opt.window_size < 366:
    opt.window_size = 366 # 366 是生成Maps/*.pt文件时的window_size
    print("\n The minimal window size should be at least 366. \n")
if opt.Playmode:
    print("\n Use UP/DOWN/LEFT/RIGHT to control the target point. \n")


if __name__ == '__main__':
    params = torch.load('Relax0.4_S0_ 2023-09-23 21_38.pt', map_location=opt.dvc)[-1]  # [-1] means the final parameters
    if not opt.KP: params[50] = 0

    env = DynamicEnv(opt, Predict_Scale_param=params[54].int())
    planner = OkayPlan(opt, params)
    while True:
        env_info = env.Map_Init()
        planner.Priori_Path_Init(env_info['start_point'], env_info['target_point']) # 为第一次规划初始化先验路径
        done = False

        while not done:
            path, collision_free = planner.plan(env_info)
            # path = [x0,x1,...,y0,y1,...], shape=(D,), on self.dvc
            # collision_free仅代表路径是否无碰撞，并不代表start_point的碰撞情况

            if opt.Playmode:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]: act = 1
                elif keys[pygame.K_UP]: act = 2
                elif keys[pygame.K_RIGHT]: act = 3
                elif keys[pygame.K_DOWN]:act = 4
                else: act = 0
                env_info = env.Update(path, act)
            else:
                env_info = env.Update(path) # act为空时，自动更新终点; act为0~4时，根据键盘输入更新终点

            done = env_info['Arrive'] + env_info['Collide']
            if env_info['Collide']: time.sleep(1)

        if opt.Record_freq >= 0:
            # 保存最后一张图并退出
            pygame.image.save(env.window, f'Records/{env.counter-1}.png')
            print(f'Timestep 0~{env.counter-1} saved in Recors.')
            break












