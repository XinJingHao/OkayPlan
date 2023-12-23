from H_SEPSO import SEPSO_UP
import argparse
import torch


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

'''Hyperparameter Setting for SEPSO_UP'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='Runing devices for SEPSO UP&DOWN')
parser.add_argument('--N', type=int, default=10, help='Number of Particles in one Group')
parser.add_argument('--T', type=int, default=300, help='Total iterations of H-SEPSO')
parser.add_argument('--write', type=str2bool, default=True, help='Whether record fitness curve')
parser.add_argument('--save', type=str2bool, default=True, help='Whether save the evolved hyperparameters')
parser.add_argument('--relax', type=float, default=0.4, help='relax of T/G/Pbest value')
parser.add_argument('--seed', type=int, default=0, help='random seed')
opt = parser.parse_args()


if __name__ == '__main__':
    sepso_up = SEPSO_UP(opt)
    params = torch.tensor([0.4, 0.7, 0.8, 0.8, 0.2, 0.9, 0.4, 0.9, # w_init
                           0.5, 0.57, 0.125, 0.75, 0.5, 0.555, 0.25, 0.333, # w_end_ratio
                           0.2, 0.1, 0.6, 0.4, 0.3, 0.5, 0.8, 0.3, # v_limit_ratio
                           2,1,2,2,2,2,1,1, # C1
                           1,1,2,2,1,1,2,2, # C2
                           1,2,1,1,2,2,2,2], dtype=torch.float,device=sepso_up.dvc) # C3

    sepso_up.reset(params)
    sepso_up.iterate()