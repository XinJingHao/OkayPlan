from torch.utils.tensorboard import SummaryWriter
from L_SEPSO import SEPSO_Down
from datetime import datetime
import os,shutil
import torch
import time



class SEPSO_UP():
    def __init__(self, opt):
        '''Use Hierarchical Self-Evolving Framework (HSEF) to tune the hyperparameters of OkayPlan autonomously'''

        self.T = opt.T
        self.dvc = torch.device(opt.dvc)
        self.G, self.N, self.D = 8, opt.N, 55 # number of Groups, particles in goups, and particle dimension
        self.arange_idx = torch.arange(self.G, device=self.dvc)  # 索引常量
        self.fitness = torch.ones((self.G, self.N), device=self.dvc) * torch.inf

        # 代表SEPSO_DOWN的w_init, w_end_rate(w_end = w_init*w_end_rate), v_limit_ratio, C1, C2, C3
        Hyper_Max_range = torch.tensor([0.9, 0.9, 0.8, 2.0, 2.0, 2.0]).unsqueeze(-1).repeat(1,self.G).view(-1) #(48,)
        Hyper_Min_range = torch.tensor([0.2, 0.1, 0.1, 1.0, 1.0, 1.0]).unsqueeze(-1).repeat(1,self.G).view(-1) #(48,)
        self.Max_range = torch.concat((Hyper_Max_range, torch.tensor([4,   6, 4,   6, 0.5, 40, 20]))).to(self.dvc) #48~54
        self.Min_range = torch.concat((Hyper_Min_range, torch.tensor([0.1, 1, 0.1, 1, 0.0, 1, 5]))).to(self.dvc) #48~54

        # R Matrix, (4,G,N,1)
        self.Random = torch.ones((4,self.G,self.N,1), device=self.dvc) #更新速度时在装载随机数

        # 生成本次实验名字
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        self.Exp_Name = f'Relax{opt.relax}_S{opt.seed}_' + timenow

        self.save = opt.save
        if opt.save:
            try: os.mkdir(f'Tbests_{self.T}_{self.D}')
            except: pass
            self.Tbest_all = torch.zeros((self.T, self.D))

        self.write = opt.write
        if opt.write:
            writepath = 'runs/'+self.Exp_Name
            if os.path.exists(writepath): shutil.rmtree(writepath)
            self.writer = SummaryWriter(log_dir=writepath)

        self.relax = opt.relax # relax = 1.0 means that relaxation strategy is not used.
        assert 0<self.relax and self.relax<=1.0


    def _uniform_random(self, low, high, shape):
        '''Generate uniformly random number in [low, high) in 'shape' on self.dvc'''
        return (high - low)*torch.rand(shape, device=self.dvc) + low

    def reset(self, params):
        '''Reset the parameters and the particles of the DTPSO'''
        # Inertia Initialization for 8 Groups
        self.w_init = params[0:self.G].unsqueeze(-1).unsqueeze(-1)
        self.w_end = (params[0:self.G]*params[self.G:(2*self.G)]).unsqueeze(-1).unsqueeze(-1)  # 0.2, 0.4, 0.1, 0.6, 0.1, 0.5, 0.1, 0.3
        self.w_delta = (self.w_init - self.w_end)/self.T # (G,1,1)


        # Velocity Constraint Initialization for 8 Groups
        v_limit_ratio = params[(2*self.G):(3*self.G)] #(G,)
        Haf_range = ((self.Max_range - self.Min_range)/2).expand((self.G, 1, self.D)) # (G,1,D), 搜索区间长度的一半,相当于DTPSO的Search_range[1]
        self.v_max = v_limit_ratio[:, None, None] * Haf_range #(G,1,D) + (G,1,1)*(G,1,D) = (G,1,D)
        self.v_min = -self.v_max #(G,1,D)


        # H Matrix, (4,G=8,1,1)
        self.Hyper = torch.ones((4,self.G), device=self.dvc)
        self.Hyper[1] = params[(3*self.G):(4*self.G)]
        self.Hyper[2] = params[(4*self.G):(5*self.G)]
        self.Hyper[3] = params[(5*self.G):(6*self.G)]
        self.Hyper.unsqueeze_(-1).unsqueeze_(-1)

        # L Matrix, (4,G,N,D)
        self.Locate = torch.zeros((4,self.G,self.N,self.D), device=self.dvc)
        self.Locate[1:4] = self._uniform_random(low=self.Min_range, high=self.Max_range, shape=(self.G,self.N,self.D)) #[0,X,X,X]

        # K Matrix, (4,G,N,D)
        self.Kinmtc = torch.zeros((4,self.G,self.N,self.D), device=self.dvc) #[V,Pbest,Gbest,Tbest]
        self.Kinmtc[0] = self._uniform_random(low=-self.Max_range, high=self.Max_range, shape=(self.G,self.N,self.D))
        self.Kinmtc[0].clip_(self.v_min, self.v_max)  # 限制速度范围

        # Best Value initialization
        self.Pbest_value = torch.ones((self.G,self.N), device=self.dvc) * torch.inf
        self.Gbest_value = torch.ones(self.G, device=self.dvc) * torch.inf
        self.Tbest_value = torch.inf


    def iterate(self):
        sepso_down = SEPSO_Down(self.dvc, Random_Obs = True, DPI=True, Kinematics_Penalty=True)

        '''双for循环遍历计算SEPSO_UP所有粒子的Fitness'''
        for i in range(self.T):
            t0 = time.time()
            for Gid in range(self.G):
                for Nid in range(self.N):
                    fitness = sepso_down.evaluate_params(self.Locate[1,Gid,Nid].clone())
                    self.fitness[Gid, Nid] = fitness
                    print(f'Evolved counter:{i}, G:{Gid}, N:{Nid}')

            ''' Step 2: 更新Pbest, Gbest, Tbest 的 value 和 particle '''
            # Pbest
            P_replace = (self.fitness < self.Pbest_value) # (G,N)
            Relaxec_P_replace = (self.fitness < (self.Pbest_value*self._uniform_random(self.relax, 1.0, (self.G,self.N)))) # (G,N)
            self.Kinmtc[1, Relaxec_P_replace] = self.Locate[1, Relaxec_P_replace] # 更新Pbest_particles
            self.Pbest_value[P_replace] = self.fitness[P_replace]  # 更新Pbest_value

            # Gbest
            values, indices = self.fitness.min(dim=-1)
            G_replace = (values < self.Gbest_value) # (G,)
            Relaxed_G_replace = (values < (self.Gbest_value * self._uniform_random(self.relax, 1.0, (self.G,))))  # (G,)
            self.Kinmtc[2, Relaxed_G_replace] = (self.Locate[2, self.arange_idx, indices][Relaxed_G_replace].unsqueeze(1)) # 更新Gbest_particles
            self.Gbest_value[G_replace] = values[G_replace]  # 更新Gbest_value

            # Tbest
            min_fitness = self.fitness.min()
            if min_fitness < self.Tbest_value * self._uniform_random(self.relax, 1.0, (1,)): # 更新Tbest_particles
                flat_idx = self.fitness.argmin()
                self.Kinmtc[3] = (self.Locate[3, flat_idx//self.N, flat_idx % self.N]).clone() #这里必须clone, 否则数据联动
            if min_fitness < self.Tbest_value:
                self.Tbest_value = min_fitness # 更新Tbest_value

            ''' Step 3: 更新速度 '''
            self.Hyper[0] = self.w_init - self.w_delta*i  # 惯性因子衰减
            self.Random[1:4] = torch.rand((3,self.G,self.N,1), device=self.dvc) #装载随机数
            self.Kinmtc[0] = (self.Hyper * self.Random * (self.Kinmtc - self.Locate)).sum(dim=0) #(G,N,D)
            self.Kinmtc[0].clip_(self.v_min, self.v_max) # 限制速度范围

            ''' Step 4: 更新位置 '''
            self.Locate[1:4] += self.Kinmtc[0] # (3,G,N,D) + (G,N,D) = (3,G,N,D)
            self.Locate[1:4].clip_(self.Min_range, self.Max_range) # 限制位置范围

            '''Print, Write, and Save'''
            MeanFit = self.fitness.mean()
            time_per_iteration = round(time.time()-t0,1)
            print(f'IterationTime:{time_per_iteration}s, RemainTime:{round((self.T-i)*time_per_iteration/3600,1)}h, MeanFit:{MeanFit}')

            if self.write:
                self.writer.add_scalar('MeanFit', MeanFit, global_step=i)

            if self.save:
                self.Tbest_all[i] = self.Kinmtc[3,0,0].clone()
                if (i+1) % 10 == 0:
                    Saved_name = f'Tbests_{self.T}_{self.D}' + self.Exp_Name + '.pt'
                    torch.save(self.Tbest_all[0:i], Saved_name)

