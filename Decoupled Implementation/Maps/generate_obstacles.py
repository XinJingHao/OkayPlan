import torch
import pygame

def main():
    segs = 6 # bounding box的边数

    pygame.init()
    pygame.display.init()
    window_size = 366
    window = pygame.display.set_mode((window_size, window_size))
    clock = pygame.time.Clock()
    map = pygame.Surface((window_size, window_size))
    idx = 0
    Segments = torch.tensor([],dtype=torch.float)
    Apexes = torch.zeros((segs,2),dtype=torch.float)

    # # 加载已保存的线段并可视化
    # Obstacle_Segments = torch.load('Obstacle_Segments.pt')
    # for _ in range(Obstacle_Segments.shape[0]):
    #     pygame.draw.line(map, (0, 0, 0), Obstacle_Segments[_,0].int().numpy(), Obstacle_Segments[_,1].int().numpy(), width=4)

    pygame.draw.line(map, (255, 0, 0), (36, 0), (36, 366), width=1)
    pygame.draw.line(map, (255, 0, 0), (336, 0), (336, 366), width=1)

    while True:
        ev = pygame.event.get()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            torch.save(Segments,f'Obstacle_Segments_S{segs}.pt')
            print(f'Obstacle_Segments.pt saved, shape={Segments.shape}')
            break

        for event in ev:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                Apexes[idx] = torch.tensor(pos)
                idx += 1

                if idx > 1:
                    pygame.draw.line(map,(0, 0, 255),Apexes[idx-2].int().numpy(),Apexes[idx-1].int().numpy(),width=1)

                if idx == segs:
                    pygame.draw.line(map, (0, 0, 255), Apexes[0].int().numpy(), Apexes[-1].int().numpy(), width=1)
                    idx = 0

                    Startpoints = Apexes.clone()
                    Endpoints = torch.zeros((segs,2))
                    Endpoints[0:(segs-1)] = Apexes[1:segs].clone()
                    Endpoints[segs-1] = Apexes[0].clone()

                    obsacle = torch.stack((Startpoints, Endpoints),dim=1) #(segs,2,2)
                    print(obsacle)
                    Segments = torch.cat((Segments,obsacle))  #(O*segs,2,2)


        window.blit(map, map.get_rect())
        pygame.event.pump()
        pygame.display.update()
        clock.tick(100)


if __name__ == '__main__':
    main()