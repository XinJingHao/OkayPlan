# OkayPlan: A real-time path planning algorithm for dynamic environments
This is the authors' implementation of [OkayPlan](https://arxiv.org/abs/2401.05019).

<div align="center">
<img width="100%" height="auto" src="https://github.com/XinJingHao/OkayPlan/blob/main/Overview.png">
</div>

<br>

## Dependencies:

```bash
torch >= 2.1.1
pygame >= 2.5.2
numpy >= 1.26.2
tensorboard >= 2.15.1

python >= 3.11.5
```
You can install **torch** by following the guidance from its [official website](https://pytorch.org/get-started/locally/). We strongly suggest you install the **CUDA 12.1** version, though CPU version or lower CUDA version are also supported.

<br>

## How to use:
After installation, you can run OkayPlan via:
```bash
python Application_Phase.py
```
Then you will see:
<div align="center">
<img width="25%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/OkayPlan/navigation.gif">
</div>
where the red point is our robot, the green point is the target, and the blue curve is the path generated by OkayPlan.

### Play with Keyboard:
You can also play with OkayPlan using your **Keyboard** (UP/DOWN/LEFT/RIGHT) :
```bash
python Application_Phase.py --Playmode True --FPS 30
```

### Basic parameters:

There are 6 parameters that could be configured when running OkayPlan:

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
 
- **Playmode (bool; default `'False'`)**:
  
  - True = Control the target point with your keyboard (UP/DOWN/LEFT/RIGHT); False = Target point moves by itself



<br>

## Comparison with other methods (real-time rendering):
<div align="center">
<img width="75%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/OkayPlan/All_gpp.gif">
</div>

<br>

## Decoupled Implementation
The decoupled implementation of OkayPlan can be found in folder _'Decoupled Implementation'_, where the ```L_SEPSO.py``` is decoupled into ```Planner.py```, ```Env.py```, and ```mian.py```.

## Citing this Project

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
