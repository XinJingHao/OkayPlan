import torch
import pygame
import os, shutil
import numpy as np

class DynamicEnv():
    def __init__(self, opt, Predict_Scale_param):
        self.Predict_Scale_param = Predict_Scale_param  # params[54].int(): 决定预测线段的放缩程度
        self.dvc = opt.dvc  # 运算平台

        self.window_size = opt.window_size
        self.Search_range = opt.Search_range  # search space of the Particles
        self.Playmode = opt.Playmode
        self.Random_Obs = opt.RO
        self.seg = opt.seg
        self.Generated_Obstacle_Segments = torch.load(f'Maps/Obstacle_Segments_S{self.seg}.pt').to(self.dvc) #(seg*O,2,2)
        self.O = self.Generated_Obstacle_Segments.shape[0] // self.seg  # 障碍物数量
        # 根据window_size对障碍物位置进行移动。注意：1) Grouped_GOS与Generated_Obstacle_Segments是联动的 2) 366是生成Maps/*.pt文件时的window_size
        Grouped_GOS = self.Generated_Obstacle_Segments.reshape(self.O,self.seg,2,2)
        Grouped_GOS += (Grouped_GOS[:,0,0,:]*(opt.window_size/366-1)).unsqueeze(1).unsqueeze(1)

        self.NP = int(opt.D / 2)  # 每个粒子所代表的路径的端点数量
        self.static_obs = 2
        self.dynamic_obs = self.O - self.static_obs
        self.Obs_X_limit = torch.tensor([0.0984,0.918], device=self.dvc)*self.window_size # [36,336] for 366
        self.Obs_Y_limit = torch.tensor(self.Search_range, device=self.dvc)
        self.Obstacle_V = opt.Obstacle_V # 每个障碍物x,y方向的最大速度
        self.Target_V = opt.Target_V # 目标点的移动速度
        self.Start_V = opt.Start_V # 被控对象的移动速度

        self._Render_Init(renderPdct=opt.KP, FPS=opt.FPS)  # 渲染初始化
        self.Record_freq = opt.Record_freq # 截图频率
        if self.Record_freq>=0:
            try: shutil.rmtree('Records')
            except: pass
            os.mkdir('Records')


    def _uniform_random(self, low, high, shape):
        '''Generate uniformly random number in [low, high) in 'shape' on self.dvc'''
        return (high - low)*torch.rand(shape, device=self.dvc) + low

    def _get_envInfo(self):
        '''抽取路径规划所需要的环境信息'''
        return dict(start_point = (self.x_start, self.y_start),
                    target_point = (self.x_target, self.y_target),
                    d2target = self.d2target,
                    Obs_Segments = self.Obs_Segments, # 障碍物bounding box线段，(4*O,2,2)
                    Flat_pdct_segments = self.Flat_pdct_segments, # 障碍物运动信息线段， (dynamic_obs*4,2,2)
                    Arrive = (self.d2target < 20), # 是否到达终点
                    Collide = self.collide) # 是否发生碰撞

    def _Render_Init(self, renderPdct, FPS):
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

        self.font = pygame.font.Font(None, 36)

    def Map_Init(self):
        self.x_start, self.y_start = 0.05*self.window_size, 0.05*self.window_size # 起点坐标, (18,18) for 366
        self.x_target, self.y_target = self.Obs_X_limit[1].item(), self.Obs_X_limit[1].item() # 终点坐标, (336,336) for 366
        if self.Playmode:
            self.y_start = np.random.randint(0.05*self.window_size, self.Obs_X_limit[1].item()) # 随机起点, (18,336) for 366
            self.y_target = np.random.randint(0.05*self.window_size, self.Obs_X_limit[1].item()) # 随机终点, (18,336) for 366
        self.d2target = ((self.x_start-self.x_target)**2 + (self.y_start-self.y_target)**2)**0.5 # distance from start to target

        # 障碍物坐标：
        # Obs_Segments, shape=(O*seg,2,2), 作用: 返回给Planner进行碰撞检测
        # Grouped_Obs_Segments, shape=(O,seg,2,2), 作用: 障碍物移动; 生成障碍物预测线; 生成障碍物箭头; 绘制障碍物
        # 注意Grouped_Obs_Segments 和 Obs_Segments 是联动的
        self.Obs_Segments = self.Generated_Obstacle_Segments.clone() # (O*4,2,2)
        self.Grouped_Obs_Segments = self.Obs_Segments.reshape(self.O,self.seg,2,2) # O个障碍物，seg条边，2个端点，2维坐标

        # 障碍物速度:
        self.Obs_V = self._uniform_random(1, self.Obstacle_V, (self.O, 1, 1, 2))  # 每个障碍物的x方向速度和y方向速度
        self.Obs_V[-self.static_obs:]*=0 # 最后static_obs个障碍物不动
        self.Obs_V[0:int(self.dynamic_obs/2)] *= -1  # 前半障碍物速度反向

        # 障碍物预测线:
        pdct_Grouped_Obs_endpoints = self.Grouped_Obs_Segments + self.Predict_Scale_param * self.Obs_V
        self.Grouped_pdct_segments = torch.stack((self.Grouped_Obs_Segments[0:self.dynamic_obs, :, 0, :],
                                                  pdct_Grouped_Obs_endpoints[0:self.dynamic_obs, :, 0, :]),
                                                 dim=2)  # (dynamic_obs,seg,2,2)
        self.Flat_pdct_segments = self.Grouped_pdct_segments.reshape(self.dynamic_obs * self.seg, 2, 2) # 用于计算OkayPlan计算交叉点个数

        # Render相关:
        # 生成障碍物速度箭头
        self.Grouped_Obs_center = self.Grouped_Obs_Segments.mean(axis=-3)[:, 0, :] # (O,2)
        self.Normed_Obs_V = torch.nn.functional.normalize(self.Obs_V, dim=-1).squeeze() # (O,2)
        self.Grouped_Obs_Vend = self.Grouped_Obs_center + 20*self.Normed_Obs_V
        # 初始化轨迹图层
        self.trajectory_pyg.fill((255,255,255))
        self.Previous_Grouped_Obs_center = None
        self.Previous_start_point = None
        self.Previous_target_point = None

        self.collide = False
        self.counter = 0

        return self._get_envInfo()


    def Update(self, path, act=None):
        self.path = path
        self._render_frame()

        ''' 更新起点、终点、障碍物, 每次得到规划路径后执行一次
            act=None时，自动更新终点; act为0~4时，根据键盘输入更新终点'''

        # 起点运动:
        if self.d2target<40: x_wp, y_wp = self.x_target, self.y_target
        else: x_wp, y_wp = self.path[1].item(), self.path[self.NP + 1].item()
        V_vector = torch.tensor([x_wp, y_wp]) - torch.tensor([self.x_start, self.y_start])
        V_vector = (torch.nn.functional.normalize(V_vector,dim=0) * self.Start_V).tolist()
        self.x_start += V_vector[0]
        self.y_start += V_vector[1]
        self.d2target = ((self.x_start-self.x_target)**2 + (self.y_start-self.y_target)**2)**0.5 # distance from start to target

        # 终点运动:
        if act is not None: # 根据键盘控制移动目标点
            if act == 0: pass
            elif act == 1: self.x_target -= self.Target_V
            elif act == 2: self.y_target -= self.Target_V
            elif act == 3: self.x_target += self.Target_V
            elif act == 4: self.y_target += self.Target_V
            self.x_target = np.clip(self.x_target, self.Obs_X_limit[1].item(), self.window_size) # 注意0.88需要大于self.Obs_X_limit[1]的系数(0.82)
            self.y_target = np.clip(self.y_target, self.Search_range[0],self.Search_range[1])
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
        pdct_Grouped_Obs_endpoints = self.Grouped_Obs_Segments + self.Predict_Scale_param * self.Obs_V  # 预测位置
        # 将当前位置和预测位置连线:
        self.Grouped_pdct_segments = torch.stack((self.Grouped_Obs_Segments[0:self.dynamic_obs, :, 0, :],
                                                  pdct_Grouped_Obs_endpoints[0:self.dynamic_obs, :, 0, :]),
                                                 dim=2)  # (dynamic_obs,seg,2,2)
        self.Flat_pdct_segments = self.Grouped_pdct_segments.reshape(self.dynamic_obs * self.seg, 2, 2)  # 用于计算OkayPlan计算交叉点个数


        # 生成障碍物速度箭头, 渲染
        self.Grouped_Obs_center = self.Grouped_Obs_Segments.mean(axis=-3)[:, 0, :] # (O,2)
        self.Normed_Obs_V = torch.nn.functional.normalize(self.Obs_V, dim=-1).squeeze() # (O,2)
        self.Grouped_Obs_Vend = self.Grouped_Obs_center + 20*self.Normed_Obs_V

        return self._get_envInfo()

    def _render_frame(self):
        '''渲染+判断start_point是否与障碍物发生碰撞
        trajectory_pyg[绘制轨迹]->map_pyg[绘制障碍物,判断碰撞,画预测线]->canvas[画路径,起点,终点]->显示'''
        
        Grouped_Obs_Segments = self.Grouped_Obs_Segments.cpu().int().numpy() # (O,seg,2,2)
        Grouped_pdct_segments = self.Grouped_pdct_segments.int().cpu().numpy() # (dynamic_obs,seg,2,2)
        Grouped_Obs_center = self.Grouped_Obs_center.int().cpu().numpy() # (O,2)
        Grouped_Obs_Vend = self.Grouped_Obs_Vend.int().cpu().numpy() # (O,2)

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
        for _ in reversed(range(self.O)):  # reversed: 先画静态障碍物，使其处于底层
            # 画障碍物
            obs_color = (25, 25, 0) if _ < self.dynamic_obs else (225, 100, 0)
            pygame.draw.polygon(self.map_pyg, obs_color, Grouped_Obs_Segments[_, :, 0, :])

        '''查看start_point是否与障碍物碰撞（在障碍物内）'''
        oc_gd_map = (pygame.surfarray.array3d(self.map_pyg)[:, :, 2]) # Blue通道在障碍物处像素值为0
        x, y = int(self.x_start), int(self.y_start)
        if (oc_gd_map[x-1:x+2, y-1:y+2]==0).all(): self.collide = True # 判断点进行膨胀再判断比障碍物腐蚀再判断快
        else: self.collide = False

        for _ in range(self.dynamic_obs):
            if self.renderPdct:
                # 画预测线
                for i in range(self.seg):
                    pygame.draw.line(self.map_pyg,(255, 0, 255),Grouped_pdct_segments[_,i,0],Grouped_pdct_segments[_,i,1],width=2)
            else:
                # 画箭头
                start, end = pygame.Vector2(Grouped_Obs_center[_,0], Grouped_Obs_center[_,1]), pygame.Vector2(Grouped_Obs_Vend[_,0], Grouped_Obs_Vend[_,1])
                direction = end - start
                arrow_points = [end, end - 12 * direction.normalize().rotate(30),end - 12 * direction.normalize().rotate(-30)]
                pygame.draw.line(self.map_pyg, (200, 200, 200), start, end, 3)
                pygame.draw.polygon(self.map_pyg, (200, 200, 200), arrow_points)
        self.canvas.blit(self.map_pyg, self.map_pyg.get_rect())

        # 画路径
        waypoints_list = self.path.cpu().view(2,self.NP).transpose(0,1).tolist() # (D,) tensor to (NP,2) list
        pygame.draw.lines(self.canvas,(0, 0, 255),False,waypoints_list,width=4)


        # 画起点、终点
        pygame.draw.circle(self.canvas, (255, 0, 0), (self.x_start, self.y_start), 5)  # 起点
        pygame.draw.circle(self.canvas, (0, 255, 0), (self.x_target, self.y_target), 5)  # 终点

        # # Lead point
        # x_wp, y_wp = self.Kinmtc[3, 0, 0, 1].item(), self.Kinmtc[3, 0, 0, self.NP + 1].item()
        # pygame.draw.circle(self.canvas, (0, 255, 255), (x_wp, y_wp), 3)  # 终点

        self.window.blit(self.canvas, self.map_pyg.get_rect())
        # 显示 Timestep
        text = self.font.render(f'Timestep:{self.counter}', True, (150, 150, 150))
        self.window.blit(text, (115, 0))


        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.FPS)

        # 保存截图
        if (self.Record_freq>=0) and (self.counter % self.Record_freq == 0):
            pygame.image.save(self.window, f'Records/{self.counter}.png')
        self.counter += 1

def str2bool(v):
    '''Fix the bool BUG for argparse: transfer string to bool'''
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1', 'T'): return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0', 'F'): return False
    else: print('Wrong Input Type!')
