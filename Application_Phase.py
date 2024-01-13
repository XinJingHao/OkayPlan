from L_SEPSO import SEPSO_Down
import argparse
import torch
import time

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1', 'T'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0', 'F'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


'''Configurations:'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='Running device of SEPSO_Down')
parser.add_argument('--RO', type=str2bool, default=True, help='Random_Obs; True = random obstacles; False = consistent obstacles')
parser.add_argument('--DPI', type=str2bool, default=True, help='True for DPI(from OkayPlan), False for PI(from SEPSO)')
parser.add_argument('--KP', type=str2bool, default=True, help='Kinematics_Penalty; True = Use Kinematics Penalty; False = Not use Kinematics Penalty')
parser.add_argument('--FPS', type=int, default=0, help='Render FPS, 0 for maximum speed')
parser.add_argument('--Playmode', type=str2bool, default=False, help='Play with keyboard: UP, DOWN, LEFT, RIGHT')
opt = parser.parse_args()
print(opt)

if __name__ == '__main__':
    dvc = torch.device(opt.dvc)
    L_SEPSO = SEPSO_Down(dvc=dvc, Random_Obs = opt.RO, DPI=opt.DPI, Kinematics_Penalty=opt.KP, Playmode=opt.Playmode)
    L_SEPSO.Render_Init(renderPdct=opt.KP, FPS=opt.FPS) # 渲染开关
    params = torch.load('Relax0.4_S0_ 2023-09-23 21_38.pt', map_location=dvc)[-1] # [-1] means the final parameters

    for seed in range(100):
        current_score, TPP = L_SEPSO.Dynamicly_Plan_And_Track(params, seed=seed)  # 这里current_score实际上是行驶路径长度(未到达时为0)
        if current_score == 0: time.sleep(1)



