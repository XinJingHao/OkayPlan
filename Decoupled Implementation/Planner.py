from collections import deque
import numpy as np
import torch

class OkayPlan():
    '''
    OkayPlan: A real-time path planner for dynamic environment
    Author: Jinghao Xin

    Paper Web: https://arxiv.org/abs/2401.05019
    Code Web: https://github.com/XinJingHao/OkayPlan

    Cite this algorithm:
    @misc{OkayPlan,
      title={OkayPlan: Obstacle Kinematics Augmented Dynamic Real-time Path Planning via Particle Swarm Optimization},
      author={Jinghao Xin and Jinwoo Kim and Shengjia Chu and Ning Li},
      year={2024},
      eprint={2401.05019},
      archivePrefix={arXiv},
      primaryClass={cs.RO}}

    Only for non-commercial purposes
    All rights reserved
    '''
    def __init__(self, opt, params):
        self.dvc = opt.dvc  # 运算平台

        '''Hyperparameter Initialization'''
        self.params = params # 该组参数的Group Number=8
        self.G = 8 # Group数量应该严格等于生成params时的种群数量
        # Inertia Initialization for 8 Groups:
        self.w_init = self.params[0:self.G].unsqueeze(-1).unsqueeze(-1)
        self.w_end = (self.params[0:self.G] * self.params[self.G:(2 * self.G)]).unsqueeze(-1).unsqueeze(-1)
        self.Max_iterations = opt.Max_iterations  # 每帧的最大迭代次数
        self.w_delta = (self.w_init - self.w_end) / self.Max_iterations  # (G,1,1)
        # Velocity Initialization for 8 Groups:
        self.v_limit_ratio = self.params[(2 * self.G):(3 * self.G)].unsqueeze(-1).unsqueeze(-1)  # (G,1,1)
        self.v_init_ratio = 0.7 * self.v_limit_ratio  # (G,1,1)
        # H Matrix, (4,G=8,1,1):
        self.Hyper = torch.ones((4, self.G), device=self.dvc)
        self.Hyper[1] = self.params[(3 * self.G):(4 * self.G)]
        self.Hyper[2] = self.params[(4 * self.G):(5 * self.G)]
        self.Hyper[3] = self.params[(5 * self.G):(6 * self.G)]
        self.Hyper.unsqueeze_(-1).unsqueeze_(-1)

        '''Particle Related'''
        self.N, self.D = opt.N, opt.D # number of Groups, particles in goups, and particle dimension
        self.arange_idx = torch.arange(self.G, device=self.dvc)  # 索引常量
        self.Search_range = opt.Search_range  # search space of the Particles
        self.Random = torch.zeros((4,self.G,self.N, 1), device=self.dvc)
        self.Kinmtc = torch.zeros((4, self.G, self.N, self.D), device=self.dvc)  # [V,Pbest,Gbest,Tbest]
        self.Locate = torch.zeros((4, self.G, self.N, self.D), device=self.dvc) # [0,X,X,X]

        '''Path Related'''
        self.NP = int(self.D/2) # 每个粒子所代表的路径的端点数量
        self.S = self.NP-1 # 每个粒子所代表的路径的线段数量
        self.P = self.G*self.N*self.S # 所有粒子所代表的路径的线段总数量
        self.rd_area = 0.5 * (self.Search_range[1] - self.Search_range[0]) / (self.S)  # SEPSO中先验知识初始化粒子时的随机范围
        self.Start_V = opt.Start_V

        '''Auto Truncation'''
        # For info, check https://arxiv.org/abs/2308.10169
        self.AT = True
        self.TrucWindow = 20
        self.std_Trsd = opt.Quality  # 自动截断判据中,std的阈值: 越小,每次规划耗时越多,但结果更好更稳定. 应在实时性和规划质量间折衷(原论文中取的10)

        '''Dynamic Prioritized Initialization'''
        self.DPI = opt.DPI

    def _uniform_random(self, low, high, shape):
        '''Generate uniformly random number in [low, high) in 'shape' on self.dvc'''
        return (high - low)*torch.rand(shape, device=self.dvc) + low

    def Priori_Path_Init(self, start, target):
        '''在起点和终点之间均匀插值生成初始先验路径，每次导航前需要执行一次，导航中无需执行'''
        Path_Xs = (start[0] + (target[0] - start[0]) * torch.arange(self.NP, device=self.dvc) / (self.NP - 1))
        Path_Ys = (start[1] + (target[1] - start[1]) * torch.arange(self.NP, device=self.dvc) / (self.NP - 1))
        self.Priori_Path = torch.cat((Path_Xs, Path_Ys))

    def _fix_apex(self):
        '''固定路径的首末端点, 注意x_start,y_start,x_target,y_target都是标量'''
        self.Locate[1:4, :, :, 0] = self.x_start
        self.Locate[1:4, :, :, self.NP] = self.y_start
        self.Locate[1:4, :, :, self.NP - 1] = self.x_target
        self.Locate[1:4, :, :, 2 * self.NP - 1] = self.y_target

    def _ReLocate(self):
        '''初始化粒子群动力学特征，每次iterate前要执行一次'''
        # Dynamic Prioritized Initialization // Prioritized Initialization
        if self.DPI:
            # OkayPlan Position Init: 根据先验知识Priori_X(上次迭代的Tbest)来初始化粒子群
            Mid_points = torch.tensor([[(self.x_start+self.x_target)/2],[(self.y_start+self.y_target)/2]],device=self.dvc).repeat((1,self.NP)).reshape(self.D) #(D,)
            self.Locate[1:4] = Mid_points + self._uniform_random(low=-self.d2target/2, high=self.d2target/2, shape=(self.G,self.N,self.D))
            # self.Locate[1:4, :, 0:self.params[53].int()] = Priori_X  # (3,G,RN,D), old version, not necessary
            self.Locate[1:4, :, 0] = self.Priori_Path  # (3,G,0,D), new version, replace params[53] for 1
        else:
            # SEPSO Position Init: From https://github.com/XinJingHao/Real-time-Path-planning-with-SEPSO
            RN = int(0.25*self.N)
            self.Locate[1:4] = self._uniform_random(low=self.Search_range[0], high=self.Search_range[1], shape=(self.G,self.N,self.D)) #[0,X,X,X]
            self.Locate[1:4, :, 0:RN] = self.Priori_Path + self._uniform_random(-self.rd_area, self.rd_area, (RN,self.D)) # (3,G,RN,D)

        # 限制粒子位置至合法范围
        self.Locate[1:4].clip_(self.Search_range[0], self.Search_range[1])
        self._fix_apex()

        # Velocity Init: (+20防止后期d2target太小时，速度太小)
        self.Kinmtc[0] = self._uniform_random(low=-(self.d2target+20), high=(self.d2target+20), shape=(self.G,self.N,self.D)) * self.v_init_ratio

        # Best Value Init:
        self.Pbest_value = torch.ones((self.G,self.N), device=self.dvc) * torch.inf
        self.Gbest_value = torch.ones(self.G, device=self.dvc) * torch.inf
        self.Tbest_value = torch.inf

        # Auto-Truncation Mechanism
        self.Tbest_Collision_Free = False
        self.Tbest_values_deque = deque(maxlen=self.TrucWindow)

        # Adaptive Velocity range
        self.v_min = -(self.v_limit_ratio * (self.d2target+20)) # (G,1,1), (+20防止后期d2target太小时，速度太小)
        self.v_max = (self.v_limit_ratio * (self.d2target+20)) # (G,1,1), (+20防止后期d2target太小时，速度太小)

    def _Cross_product_for_VectorSet(self, V_PM, V_PP):
        '''计算 向量集V_PM 和  向量集V_PP 的交叉积 (x0*y1-x1*y0)
            V_PM = torch.tensor((p, m, 2, 2))
            V_PP = torch.tensor((p,2))
            Output = torch.tensor((p, m, 2))'''
        return V_PM[:, :, :, 0] * V_PP[:, 1, None, None] - V_PM[:, :, :, 1] * V_PP[:, 0, None, None]

    def _Is_Seg_Ingersection_PtoM(self, P, M):
        '''利用[交叉积-跨立实验]判断 线段集P 与 线段集M 的相交情况
            P = torch.tensor((p,2,2))
            M = torch.tensor((m,2,2))
            Output = torch.tensor((p,m)), dtype=torch.bool'''

        V_PP = P[:, 1] - P[:, 0]  # (p, 2),自身向量
        V_PM = M - P[:, None, None, 0]  # (p, m, 2, 2), 自身起点与其他线段端点构成的向量
        Flag1 = self._Cross_product_for_VectorSet(V_PM, V_PP).prod(dim=-1) < 0  # (p, m)

        V_MM = M[:, 1] - M[:, 0]  # (m, 2)
        V_MP = P - M[:, None, None, 0]  # (m, p, 2, 2)
        Flag2 = self._Cross_product_for_VectorSet(V_MP, V_MM).prod(dim=-1) < 0  # (m, p)
        return Flag1 * Flag2.T

    def _get_Obs_Penalty(self):
        # 将粒子群转化为线段，并展平为(G*N*S,2,2)
        particles = self.Locate[1].clone()  # (G,N,D)
        start_points = torch.stack((particles[:,:,0:(self.NP-1)], particles[:,:,self.NP:(2*self.NP-1)]), dim=-1) #(G,N,S,2), S条线段的起点坐标
        end_points = torch.stack((particles[:,:,1:self.NP], particles[:,:,(self.NP+1):2*self.NP]), dim=-1) #(G,N,S,2), S条线段的终点坐标
        Segments = torch.stack((start_points, end_points),dim=-2) #(G,N,S,2,2), (G,N,S)条线段的端点坐标
        flatted_Segments = Segments.reshape((self.P,2,2)) # (G*N*S,2,2), 将所有粒子展平为G*N*S条线段

        # 将展平后的粒子群线段 与 地图中的障碍物边界线段 进行跨立实验，得到交叉矩阵
        Intersect_Matrix = self._Is_Seg_Ingersection_PtoM(flatted_Segments, self.Obs_Segments) # (P,M)
        Current_Obs_penalty = Intersect_Matrix.sum(dim=-1).reshape((self.G,self.N,self.S)).sum(dim=-1) #(G,N)

        # 将展平后的粒子群线段 与 障碍物运动轨迹线段 进行跨立实验，得到交叉矩阵
        Pdct_Intersect_Matrix = self._Is_Seg_Ingersection_PtoM(flatted_Segments, self.Flat_pdct_segments) # (P,M)
        Pdct_Obs_penalty = Pdct_Intersect_Matrix.sum(dim=-1).reshape((self.G, self.N, self.S)).sum(dim=-1)  # (G,N)
        return Current_Obs_penalty, Pdct_Obs_penalty

    def _get_fitness(self):
        Segments_lenth = torch.sqrt((self.Locate[1,:,:,0:(self.NP-1)] - self.Locate[1,:,:,1:self.NP]).pow(2) +
                             (self.Locate[1,:,:,self.NP:(2*self.NP-1)] - self.Locate[1,:,:,(self.NP+1):(2*self.NP)]).pow(2)) # (G,N,S)
        Path_lenth = Segments_lenth.sum(dim=-1) # (G,N) # 总路径长度
        LeadSeg_lenth = Segments_lenth[:,:,0]  # (G,N) # Start point 和 Lead point 之间的长度
        self.Obs_penalty, Pdct_Obs_penalty = self._get_Obs_Penalty() # both in (G,N)

        # Fitness = Length_Term + Collision_Penalty + Kinematics_Penalty + Lead_Point_Penalty
        # 用d2target实现Dynamic Normalization,保证不同地图尺寸时, 惩罚相都与目标相的幅值匹配
        # Note that Lead_Point_Penalty is not necessary when there is a Local Planner
        return Path_lenth +\
               self.params[48] * self.d2target * (self.Obs_penalty) ** self.params[49] + \
               self.params[50] * self.d2target * (Pdct_Obs_penalty) ** self.params[51] + \
               self.params[52] * self.d2target * (LeadSeg_lenth < 1.5*self.Start_V)

    def _iterate(self):
        ''' 粒子迭代, 规划路径( 规划结果=self.Kinmtc[3,0,0] ) '''

        ''' Step 0: 动态先验初始化'''
        self._ReLocate()

        for i in range(self.Max_iterations):
            if self.AT and (i>0.2*self.TrucWindow) and self.Tbest_Collision_Free and (np.std(self.Tbest_values_deque)<self.std_Trsd):
                break

            ''' Step 1: 计算Fitness (G,N)'''
            fitness = self._get_fitness()

            ''' Step 2: 更新Pbest, Gbest, Tbest 的 value 和 particle '''
            # Pbest
            P_replace = (fitness < self.Pbest_value) # (G,N)
            self.Pbest_value[P_replace] = fitness[P_replace] # 更新Pbest_value
            self.Kinmtc[1, P_replace] = self.Locate[1, P_replace] # 更新Pbest_particles

            # Gbest
            values, indices = fitness.min(dim=-1)
            G_replace = (values < self.Gbest_value) # (G,)
            self.Gbest_value[G_replace] = values[G_replace] # 更新Gbest_value
            self.Kinmtc[2, G_replace] = (self.Locate[2, self.arange_idx, indices][G_replace].unsqueeze(1))

            # Tbest
            flat_idx = fitness.argmin()
            idx_g, idx_n = flat_idx//self.N, flat_idx%self.N
            min_fitness = fitness[idx_g, idx_n]
            if min_fitness < self.Tbest_value:
                self.Tbest_value = min_fitness # 更新Tbest_value
                self.Kinmtc[3] = (self.Locate[3, idx_g, idx_n]).clone() #这里必须clone, 否则数据联动
                # 查看Tbest所代表的路径是否collision free:
                if self.Obs_penalty[idx_g, idx_n] == 0: self.Tbest_Collision_Free = True
                else: self.Tbest_Collision_Free = False
            self.Tbest_values_deque.append(self.Tbest_value.item())


            ''' Step 3: 更新粒子速度 '''
            self.Hyper[0] = self.w_init - self.w_delta*i  # 惯性因子衰减
            self.Random[1:4] = torch.rand((3,self.G,self.N,1), device=self.dvc) #装载随机数
            self.Kinmtc[0] = (self.Hyper * self.Random * (self.Kinmtc - self.Locate)).sum(dim=0) #(G,N,D)
            self.Kinmtc[0].clip_(self.v_min, self.v_max) # 限制速度范围

            ''' Step 4: 更新粒子位置 '''
            self.Locate[1:4] += self.Kinmtc[0] # (3,G,N,D) + (G,N,D) = (3,G,N,D)
            self.Locate[1:4].clip_(self.Search_range[0], self.Search_range[1]) # 限制位置范围
            self._fix_apex()


    def plan(self, env_info):
        ''' 获取环境信息 -> 迭代求解路径 -> 更新下一时刻的先验路径  -> 返回当前时刻路径'''

        self.x_start, self.y_start = env_info['start_point']
        self.x_target, self.y_target = env_info['target_point']
        self.d2target = env_info['d2target']
        self.Obs_Segments = env_info['Obs_Segments'] # (4*O,2,2)
        self.Flat_pdct_segments = env_info['Flat_pdct_segments'] # (dynamic_obs*4,2,2)

        self._iterate()

        self.Priori_Path = self.Kinmtc[3,0,0].clone()

        return self.Kinmtc[3,0,0].clone(), self.Tbest_Collision_Free


