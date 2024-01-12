# OkayPlan: A real-time path palnning algorithm for dynamic environments
<div align="center">
<img width="100%" height="auto" src="https://github.com/XinJingHao/OkayPlan/blob/main/Overview.png">
</div>

<br>

## Comparison with other method (real-time rendering):
<div align="center">
<img width="75%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/OkayPlan/All_gpp.gif">
</div>

## Dependencies:

```bash
torch >= 2.1.1
pygame >= 2.5.2
numpy >= 1.26.2
tensorboard >= 2.15.1

python >= 3.11.5
```
You can install **torch** by following the guidance from its [official website](https://pytorch.org/get-started/locally/). We strongly suggest you install the **CUDA 12.1** version, though CPU version or lower CUDA version are also supported.

## How to use:
After installation, you can play with OkayPlan via:
```bash
python Application_Phase.py
```
Then you will see:
<div align="center">
<img width="25%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/OkayPlan/navigation.gif">
</div>

### Basic parameters:

There are 5 parameters that could be configured when running OkayPlan:

- **dvc (string; default `'dvc'`)**:
  
  - The device that runs the OkayPlan
    
  - Should be one of `'cpu'`/`'cuda'`. We suggest using `'cuda'` (GPU) to accelerate the simulation
 
- **RO (bool; default `'True'`)**:
  
  - True = random obstacles; False = consistent obstacles

- **DPI (bool; default `'True'`)**:
  
  - True = Dynamic Prioritized Initialization; False = Prioritized Initialization

- **KP (bool; default `'True'`)**:
  
  - True = Use Kinematics Penalty; False = Not use Kinematics Penalty

- **FPS (int; default `'0'`)**:
  
  - The FPS when rendering
  - **'0'** means render at the maximum speed

You can try other configurations in the following way:
```bash
python Application_Phase.py --dvc cpu --KP False --FPS 30
```

## Citing the Project

To cite this repository in publications:

```bibtex
@misc{OkayPlan,
      title={OkayPlan: Obstacle Kinematics Augmented Dynamic Real-time Path Planning via Particle Swarm Optimization}, 
      author={Jinghao Xin and Jinwoo Kim and Shengjia Chu and Ning Li},
      year={2024},
      eprint={2401.05019},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
