import torch
import pygame
import os

def main():
    pygame.init()
    pygame.display.init()
    window_size = 366
    window = pygame.display.set_mode((window_size, window_size))
    clock = pygame.time.Clock()
    map = pygame.Surface((window_size, window_size))
    idx = 0
    Segments = torch.tensor([],dtype=torch.float)
    Apexes = torch.zeros((4,2),dtype=torch.float)

    # # 加载已保存的线段并可视化
    # Obstacle_Segments = torch.load('Obstacle_Segments.pt')
    # for _ in range(Obstacle_Segments.shape[0]):
    #     pygame.draw.line(map, (0, 0, 0), Obstacle_Segments[_,0].int().numpy(), Obstacle_Segments[_,1].int().numpy(), width=4)

    pygame.draw.line(map, (255, 0, 0), (36, 0), (36, 366), width=2)
    pygame.draw.line(map, (255, 0, 0), (336, 0), (336, 366), width=2)

    while True:
        ev = pygame.event.get()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            torch.save(Segments,'Obstacle_Segments.pt')
            break

        for event in ev:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                Apexes[idx] = torch.tensor(pos)
                idx += 1

                if idx > 1:
                    pygame.draw.line(map,(0, 0, 255),Apexes[idx-2].int().numpy(),Apexes[idx-1].int().numpy(),width=4)

                if idx == 4:
                    pygame.draw.line(map, (0, 0, 255), Apexes[0].int().numpy(), Apexes[-1].int().numpy(), width=4)
                    idx = 0

                    Startpoints = Apexes.clone()
                    Endpoints = torch.zeros((4,2))
                    Endpoints[0:3] = Apexes[1:4].clone()
                    Endpoints[3] = Apexes[0].clone()

                    obsacle = torch.stack((Startpoints, Endpoints),dim=1) #(4,2,2)
                    print(obsacle)
                    Segments = torch.cat((Segments,obsacle))  #(O*4,2,2)


        window.blit(map, map.get_rect())
        pygame.event.pump()
        pygame.display.update()
        clock.tick(100)


if __name__ == '__main__':
    main()