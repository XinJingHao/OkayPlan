from collections import deque
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import torch
import numpy as np
import time

class SEPSO_Down():
    def __init__(self, dvc, Random_Obs = True, DPI=True, Kinematics_Penalty=True, Playmode=False):
        '''
        dvc(string; 'cpu' or 'cuda'): running device of SEPSO_Down
        Random_Obs(bool; True or False): True = random obstacles; False = consistent obstacles
        DPI(bool; True or False): True = Dynamic Prioritized Initialization; False = Prioritized Initialization
        Kinematics_Penalty(bool; True or False): True = Use Kinematics Penalty; False = Not use Kinematics Penalty
        '''

        # Particle Related:
        self.dvc = dvc
        self.Max_iterations = 50 # 每帧的最大迭代次数
        self.G, self.N, self.D = 8, 170, 20 # number of Groups, particles in goups, and particle dimension
        self.arange_idx = torch.arange(self.G, device=self.dvc)  # 索引常量
        self.Search_range = (5., 360.)  # search space of the Particles
        self.Random = torch.zeros((4,self.G,self.N,1), device=self.dvc)
        self.Kinmtc = torch.zeros((4, self.G, self.N, self.D), device=self.dvc)  # [V,Pbest,Gbest,Tbest]
        self.Locate = torch.zeros((4, self.G, self.N, self.D), device=self.dvc) # [0,X,X,X]

        # Map Related:
        self.window_size = 366
        self.Playmode = Playmode
        self.Random_Obs = Random_Obs
        if self.Random_Obs:
            self.Generated_Obstacle_Segments = torch.load('Maps/RMO_Obstacle_Segments.pt').to(self.dvc) #(M,2,2) or (4*O,2,2)
        else:
            self.Generated_Obstacle_Segments = torch.load('Maps/CMO_Obstacle_Segments.pt').to(self.dvc) #(M,2,2) or (4*O,2,2)
        self.O = self.Generated_Obstacle_Segments.shape[0] // 4  # 障碍物数量
        self.static_obs = 2
        self.dynamic_obs = self.O - self.static_obs
        self.Obs_X_limit = torch.tensor([46,300], device=self.dvc)
        self.Obs_Y_limit = torch.tensor([0, 366], device=self.dvc)
        self.Obstacle_V = 4 # 每个障碍物x,y方向的最大速度
        self.Target_V = 3
        self.Start_V = 6

        # Path Related:
        self.NP = int(self.D/2) # 每个粒子所代表的路径的端点数量
        self.S = self.NP-1 # 每个粒子所代表的路径的线段数量
        self.P = self.G*self.N*self.S # 所有粒子所代表的路径的线段总数量
        Init_X = (20 + (330 - 20) * torch.arange(self.NP, device=self.dvc) / (self.NP - 1))
        self.Init_X = torch.cat((Init_X, Init_X)) #第一次初始化时的先验位置
        self.rd_area = 0.5 * (self.Search_range[1] - self.Search_range[0]) / (self.S)  # SEPSO中先验知识初始化粒子时的随机范围

        # Auto Truncation:
        # For info, check https://arxiv.org/abs/2308.10169
        self.AT = True
        self.TrucWindow = 20

        # Dynamic Prioritized Initialization
        self.DPI = DPI

        # Obstacle kinematics augmented optimization problem
        self.Kinematics_Penalty = Kinematics_Penalty


        if self.Random_Obs: print('----------------- Using Randomly Moving Obstacles ------------------')
        else: print('----------------- Using Consistently Moving Obstacles ------------------')


        self.render = False

    def _uniform_random(self, low, high, shape):
        '''Generate uniformly random number in [low, high) in 'shape' on self.dvc'''
        return (high - low)*torch.rand(shape, device=self.dvc) + low

    def _HyperParameter_Init(self):
        '''SEPSO的48个超参数初始化, 在__init__中执行一次即可'''
        # Inertia Initialization for 8 Groups
        self.w_init = self.params[0:self.G].unsqueeze(-1).unsqueeze(-1)
        self.w_end = (self.params[0:self.G]*self.params[self.G:(2*self.G)]).unsqueeze(-1).unsqueeze(-1)
        self.w_delta = (self.w_init - self.w_end)/self.Max_iterations # (G,1,1)

        # Velocity Initialization for 8 Groups
        self.v_limit_ratio = self.params[(2*self.G):(3*self.G)].unsqueeze(-1).unsqueeze(-1) # (G,1,1)
        self.v_init_ratio = 0.7 * self.v_limit_ratio # (G,1,1)

        # H Matrix, (4,G=8,1,1)
        self.Hyper = torch.ones((4,self.G), device=self.dvc)
        self.Hyper[1] = self.params[(3*self.G):(4*self.G)]
        self.Hyper[2] = self.params[(4*self.G):(5*self.G)]
        self.Hyper[3] = self.params[(5*self.G):(6*self.G)]
        self.Hyper.unsqueeze_(-1).unsqueeze_(-1)

    def _Map_Init(self):
        '''每次Dynamicly_Plan_And_Track前要执行一次'''
        self.x_start, self.y_start = 20, 20 # 起点坐标
        self.x_target, self.y_target = 330, 330 # 终点坐标
        if self.Playmode:
            self.y_start = np.random.randint(20,340) # 随机起点
            self.y_target = np.random.randint(20, 340) # 随机终点
        self.d2target = ((self.x_start-self.x_target)**2 + (self.y_start-self.y_target)**2)**0.5 # distance from start to target
        self.initail_d2target = ((self.x_start-self.x_target)**2 + (self.y_start-self.y_target)**2)**0.5

        # 障碍物坐标：
        self.Obs_Segments = self.Generated_Obstacle_Segments.clone()
        self.Grouped_Obs_Segments = self.Obs_Segments.reshape(self.O,4,2,2) #注意Grouped_Obs_Segments 和 Obs_Segments 是联动的


        # 障碍物速度:
        self.Obs_V = self._uniform_random(1, self.Obstacle_V, (self.O, 1, 1, 2))  # 每个障碍物的x方向速度和y方向速度
        self.Obs_V[-self.static_obs:]*=0 # 最后static_obs个障碍物不动
        self.Obs_V[0:int(self.dynamic_obs/2)] *= -1  # 前半障碍物速度反向

        # 障碍物预测线:
        pdct_Grouped_Obs_endpoints = self.Grouped_Obs_Segments + self.params[54].int() * self.Obs_V
        self.Grouped_pdct_segments = torch.stack((self.Grouped_Obs_Segments[0:self.dynamic_obs, :, 0, :],
                                                  pdct_Grouped_Obs_endpoints[0:self.dynamic_obs, :, 0, :]),
                                                 dim=2)  # (dynamic_obs,4,2,2)

        if self.render:
            # 生成障碍物速度箭头
            self.Grouped_Obs_center = self.Grouped_Obs_Segments.mean(axis=-3)[:, 0, :] # (O,2)
            self.Normed_Obs_V = (self.Obs_V/((self.Obs_V**2).sum(dim=-1).pow(0.5).unsqueeze(dim=-1)+1e-8)).squeeze() # (O,2)
            self.Grouped_Obs_Vend = self.Grouped_Obs_center + 20*self.Normed_Obs_V

            # 初始化轨迹图层
            self.trajectory_pyg.fill((255,255,255))
            self.Previous_Grouped_Obs_center = None
            self.Previous_start_point = None
            self.Previous_target_point = None

    def _ReLocate(self, Priori_X):
        '''初始化粒子群动力学特征，每次iterate前要执行一次'''
        # Dynamic Prioritized Initialization // Prioritized Initialization
        if self.DPI:
            # OkayPlan Position Init: 根据先验知识Priori_X(上次迭代的Tbest)来初始化粒子群
            Mid_points = torch.tensor([[(self.x_start+self.x_target)/2],[(self.y_start+self.y_target)/2]],device=self.dvc).repeat((1,self.NP)).reshape(self.D) #(D,)
            self.Locate[1:4] = Mid_points + self._uniform_random(low=-self.d2target/2, high=self.d2target/2, shape=(self.G,self.N,self.D))
            # self.Locate[1:4, :, 0:self.params[53].int()] = Priori_X  # (3,G,RN,D), old version, not necessary
            self.Locate[1:4, :, 0] = Priori_X  # (3,G,0,D), new version, replace params[53] for 1
        else:
            # SEPSO Position Init: From https://github.com/XinJingHao/Real-time-Path-planning-with-SEPSO
            RN = int(0.25*self.N)
            self.Locate[1:4] = self._uniform_random(low=self.Search_range[0], high=self.Search_range[1], shape=(self.G,self.N,self.D)) #[0,X,X,X]
            self.Locate[1:4, :, 0:RN] = Priori_X + self._uniform_random(-self.rd_area, self.rd_area, (RN,self.D)) # (3,G,RN,D)

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
        self.Tbest_Free = False
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
        Flat_pdct_segments = self.Grouped_pdct_segments.reshape(self.dynamic_obs*4,2,2)
        Pdct_Intersect_Matrix = self._Is_Seg_Ingersection_PtoM(flatted_Segments, Flat_pdct_segments) # (P,M)
        Pdct_Obs_penalty = Pdct_Intersect_Matrix.sum(dim=-1).reshape((self.G, self.N, self.S)).sum(dim=-1)  # (G,N)
        return Current_Obs_penalty, Pdct_Obs_penalty

    def _get_fitness(self):
        Segments_lenth = torch.sqrt((self.Locate[1,:,:,0:(self.NP-1)] - self.Locate[1,:,:,1:self.NP]).pow(2) +
                             (self.Locate[1,:,:,self.NP:(2*self.NP-1)] - self.Locate[1,:,:,(self.NP+1):(2*self.NP)]).pow(2)) # (G,N,S)
        Path_lenth = Segments_lenth.sum(dim=-1) # (G,N) # 总路径长度
        LeadSeg_lenth = Segments_lenth[:,:,0]  # (G,N) # Start point 和 Lead point 之间的长度
        self.Obs_penalty, Pdct_Obs_penalty = self._get_Obs_Penalty() # both in (G,N)

        # Fitness = Length_Term + Collision_Penalty + Kinematics_Penalty + Lead_Point_Penalty
        # Note that Lead_Point_Penalty is not necessary when there is a Local Planner
        return Path_lenth +\
               self.params[48] * self.d2target * (self.Obs_penalty) ** self.params[49] + \
               self.params[50] * self.d2target * (Pdct_Obs_penalty) ** self.params[51] + \
               self.params[52] * self.d2target * (LeadSeg_lenth < 1.5*self.Start_V)

    def iterate(self):
        for i in range(self.Max_iterations):
            if self.AT and (i>0.2*self.TrucWindow) and self.Tbest_Free and (np.std(self.Tbest_values_deque)<10):
                return self.Tbest_value.item(), (i+1) # Path Lenth, Iterations per Planning

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
                if self.Obs_penalty[idx_g, idx_n] == 0: self.Tbest_Free = True
                else: self.Tbest_Free = False
            self.Tbest_values_deque.append(self.Tbest_value.item())


            ''' Step 3: 更新速度 '''
            self.Hyper[0] = self.w_init - self.w_delta*i  # 惯性因子衰减
            self.Random[1:4] = torch.rand((3,self.G,self.N,1), device=self.dvc) #装载随机数
            self.Kinmtc[0] = (self.Hyper * self.Random * (self.Kinmtc - self.Locate)).sum(dim=0) #(G,N,D)
            self.Kinmtc[0].clip_(self.v_min, self.v_max) # 限制速度范围

            ''' Step 4: 更新位置 '''
            self.Locate[1:4] += self.Kinmtc[0] # (3,G,N,D) + (G,N,D) = (3,G,N,D)
            self.Locate[1:4].clip_(self.Search_range[0], self.Search_range[1]) # 限制位置范围
            self._fix_apex()

        return self.Tbest_value.item(), (i + 1)  # Path Lenth, Iterations per Planning

    def _fix_apex(self):
        '''固定路径的首末端点, 注意x_start,y_start,x_target,y_target都是标量'''
        self.Locate[1:4, :, :, 0] = self.x_start
        self.Locate[1:4, :, :, self.NP] = self.y_start
        self.Locate[1:4, :, :, self.NP - 1] = self.x_target
        self.Locate[1:4, :, :, 2 * self.NP - 1] = self.y_target

    def _update_map(self, act=None):
        ''' 更新起点、终点、障碍物, 每次iterate后执行一次
            act=None时，自动更新终点; act为0~4时，根据键盘输入更新终点'''
        # 起点运动:
        if self.d2target<40: x_wp, y_wp = self.x_target, self.y_target
        else: x_wp, y_wp = self.Kinmtc[3, 0, 0, 1].item(), self.Kinmtc[3, 0, 0, self.NP + 1].item()
        V_vector = torch.tensor([x_wp, y_wp]) - torch.tensor([self.x_start, self.y_start])
        V_vector = V_vector / torch.sqrt(V_vector.pow(2).sum()+1e-6) # normalization
        self.x_start += V_vector[0].item() * self.Start_V
        self.y_start += V_vector[1].item() * self.Start_V
        self.d2target = ((self.x_start-self.x_target)**2 + (self.y_start-self.y_target)**2)**0.5 # distance from start to target

        # 终点运动:
        if act is not None: # 根据键盘控制移动目标点
            if act == 0: pass
            elif act == 1: self.x_target -= self.Target_V
            elif act == 2: self.y_target -= self.Target_V
            elif act == 3: self.x_target += self.Target_V
            elif act == 4: self.y_target += self.Target_V
            self.x_target = np.clip(self.x_target, 320, 366)
            self.y_target = np.clip(self.y_target, 0,366)
        else: # 沿着y轴自动移动目标点
            self.y_target += self.Target_V
            if self.y_target > self.Search_range[1]:
                self.y_target = self.Search_range[1]
                self.Target_V *= -1
            if self.y_target < self.Search_range[0]:
                self.y_target = self.Search_range[0]
                self.Target_V *= -1

        # 障碍物运动
        self.Grouped_Obs_Segments += self.Obs_V
        Flag_Vx = ((self.Grouped_Obs_Segments[:, :, :, 0] < self.Obs_X_limit[0]) |
                   (self.Grouped_Obs_Segments[:, :, :, 0] > self.Obs_X_limit[1])).any(dim=-1).any(dim=-1)
        self.Obs_V[Flag_Vx, :, :, 0] *= -1
        Flag_Vy = ((self.Grouped_Obs_Segments[:, :, :, 1] < self.Obs_Y_limit[0]) |
                   (self.Grouped_Obs_Segments[:, :, :, 1] > self.Obs_Y_limit[1])).any(dim=-1).any(dim=-1)
        self.Obs_V[Flag_Vy, :, :, 1] *= -1

        # 随机障碍物
        if self.Random_Obs:
            self.Obs_V += self._uniform_random(-1, 1, (self.O, 1, 1, 2))  # 每个障碍物的x方向速度和y方向速度
            self.Obs_V[-self.static_obs:]*=0 # 最后static_obs个障碍物不动
            self.Obs_V.clip_(-self.Obstacle_V, self.Obstacle_V) # 限速

        # 生成障碍物运动预测线
        pdct_Grouped_Obs_endpoints = self.Grouped_Obs_Segments + self.params[54].int() * self.Obs_V  # 预测位置
        # 将当前位置和预测位置连线:
        self.Grouped_pdct_segments = torch.stack((self.Grouped_Obs_Segments[0:self.dynamic_obs, :, 0, :],
                                                  pdct_Grouped_Obs_endpoints[0:self.dynamic_obs, :, 0, :]),
                                                 dim=2)  # (dynamic_obs,4,2,2)

        if self.render:
            # 生成障碍物速度箭头
            self.Grouped_Obs_center = self.Grouped_Obs_Segments.mean(axis=-3)[:, 0, :] # (O,2)
            self.Normed_Obs_V = (self.Obs_V/((self.Obs_V**2).sum(dim=-1).pow(0.5).unsqueeze(dim=-1)+1e-8)).squeeze() # (O,2)
            self.Grouped_Obs_Vend = self.Grouped_Obs_center + 20*self.Normed_Obs_V




    def Render_Init(self, renderPdct, FPS):
        self.render = True

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()
        self.canvas = pygame.Surface((self.window_size, self.window_size))

        self.map_pyg = pygame.Surface((self.window_size, self.window_size)) # 障碍物图层
        self.trajectory_pyg = pygame.Surface((self.window_size, self.window_size)) # 轨迹图层

        # Prediction Related:
        self.renderPdct = renderPdct # True: 画预测线, False: 画障碍物速度方向

        self.FPS = FPS

    def _render_frame(self):
        Grouped_Obs_Segments = self.Grouped_Obs_Segments.cpu().int().numpy()
        Grouped_pdct_segments = self.Grouped_pdct_segments.int().cpu().numpy()
        Grouped_Obs_center = self.Grouped_Obs_center.int().cpu().numpy()
        Grouped_Obs_Vend = self.Grouped_Obs_Vend.int().cpu().numpy()

        # 绘制轨迹图层：
        if self.Previous_Grouped_Obs_center is not None:
            for _ in range(self.dynamic_obs):
                pygame.draw.line(self.trajectory_pyg,(200, 200, 200),self.Previous_Grouped_Obs_center[_], Grouped_Obs_center[_],width=5) # 障碍物轨迹
            pygame.draw.line(self.trajectory_pyg,(255, 174, 185),self.Previous_start_point, (self.x_start, self.y_start),width=5) # 起点轨迹
            pygame.draw.line(self.trajectory_pyg,(202, 255, 112),self.Previous_target_point, (self.x_target, self.y_target),width=5) # 终点轨迹

        self.Previous_Grouped_Obs_center = Grouped_Obs_center.copy()
        self.Previous_start_point = (self.x_start, self.y_start)
        self.Previous_target_point = (self.x_target, self.y_target)

        # 绘制障碍物图层：
        self.map_pyg.blit(self.trajectory_pyg, self.trajectory_pyg.get_rect())
        for _ in range(self.O):
            # 画障碍物
            obs_color = (50, 50, 50) if _ < self.dynamic_obs else (225, 100, 0)
            pygame.draw.polygon(self.map_pyg, obs_color, Grouped_Obs_Segments[_, :, 0, :])

            # 画预测线
            if _ < self.dynamic_obs:
                if self.renderPdct:
                    for i in range(4):
                        pygame.draw.line(self.map_pyg,(255, 0, 255),Grouped_pdct_segments[_,i,0],Grouped_pdct_segments[_,i,1],width=2)
                else:
                    start, end = pygame.Vector2(Grouped_Obs_center[_,0], Grouped_Obs_center[_,1]), pygame.Vector2(Grouped_Obs_Vend[_,0], Grouped_Obs_Vend[_,1])
                    direction = end - start
                    arrow_points = [end, end - 12 * direction.normalize().rotate(30),end - 12 * direction.normalize().rotate(-30)]
                    pygame.draw.line(self.map_pyg, (200, 200, 200), start, end, 3)
                    pygame.draw.polygon(self.map_pyg, (200, 200, 200), arrow_points)
        self.canvas.blit(self.map_pyg, self.map_pyg.get_rect())

        # 画路径
        for p in range(self.NP - 1):
            pygame.draw.line(
                self.canvas,
                (0, 0, 255),
                (self.Kinmtc[3, 0, 0, p].item(), self.Kinmtc[3, 0, 0, p + self.NP].item()),
                (self.Kinmtc[3, 0, 0, p + 1].item(), self.Kinmtc[3, 0, 0, p + self.NP + 1].item()),
                width=4)


        # 画起点、终点
        pygame.draw.circle(self.canvas, (255, 0, 0), (self.x_start, self.y_start), 5)  # 起点
        pygame.draw.circle(self.canvas, (0, 255, 0), (self.x_target, self.y_target), 5)  # 终点

        # # Lead point
        # x_wp, y_wp = self.Kinmtc[3, 0, 0, 1].item(), self.Kinmtc[3, 0, 0, self.NP + 1].item()
        # pygame.draw.circle(self.canvas, (0, 255, 255), (x_wp, y_wp), 3)  # 终点

        self.window.blit(self.canvas, self.map_pyg.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.FPS)

    def Dynamicly_Plan_And_Track(self, params, seed=None):
        '''For Render and Evaluation'''
        self.params = params
        if not self.Kinematics_Penalty: self.params[50] = 0

        # Seed Everything
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self._Map_Init()
        self._HyperParameter_Init()
        c, TPP = 0, 0
        while True:
            if c == 0: self._ReLocate(self.Init_X.clone())
            else: self._ReLocate(self.Kinmtc[3,0,0].clone())

            c += 1
            t0 = time.time()
            pl, ipp = self.iterate() # Path Lenth, Iterations per Planning
            TPP = TPP + ((time.time()-t0) - TPP) / c #增量法求平均 Time Per Planning

            if self.render: self._render_frame()

            if self.Playmode:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]: act = 1
                elif keys[pygame.K_UP]: act = 2
                elif keys[pygame.K_RIGHT]: act = 3
                elif keys[pygame.K_DOWN]:act = 4
                else: act = 0
                self._update_map(act) # 当act不为空时，根据键盘输入控制目标点
            else:
                self._update_map() # 当act为空时，自动更新目标点

            if self.d2target < 20:
                print(f'Arrived! Lenth:{c*self.Start_V}')
                return c*self.Start_V, TPP  # 总行驶路程


            if not self.Tbest_Free:
                print('Collide!')
                return 0, TPP


    def evaluate_params(self, params, seed=None):
        '''用于Self-evolve'''
        self.params = params
        if not self.Kinematics_Penalty: self.params[50] = 0
        # 0~7: w_init
        # 8~15: w_decay
        # 16~23: v_limit_ratio
        # 24~31: C1
        # 32~39: C2
        # 40~47: C3
        # 48~52: Fitness function
        # 53: gamma(PI interval). Discard.
        # 54: Predict lenth(预测线长度)

        # Seed Everything
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self._Map_Init()
        self._HyperParameter_Init()
        for c in range(150):
            if c == 0: self._ReLocate(self.Init_X.clone())
            else: self._ReLocate(self.Kinmtc[3, 0, 0].clone())

            self.iterate() #为当前帧进行规划
            self._update_map() # 更新地图中的障碍物、起点、终点


            if self.d2target < 20: # 抵达终点
                return -1 / (c * self.Start_V)  # -1/总行驶路程

            if not self.Tbest_Free: # 撞墙
                return 0

        return 0 # Did not reach the Target point after 150 steps
