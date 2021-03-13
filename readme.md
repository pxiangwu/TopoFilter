## A Topological Filter for Learning with Label Noise (NeurIPS 2020, [Paper](https://proceedings.neurips.cc/paper/2020/file/f4e3ce3e7b581ff32e40968298ba013d-Paper.pdf))

![](https://github.com/pxiangwu/TopoFilter/blob/master/figs/intro.png)

## Requirements
- PyTorch 0.4.1 (have not tested on other versions)
- Python 3.6 (for the purpose of compiling C++ code. Other 3.x versions should also work.)
- scipy 1.1.0 (this is due to the computation of distribution mode)
- termcolor, etc (which can be easily installed with pip)

## Usage
- Compile the C++ code for computing the connected components. In folder `ref`, run `./compile_pers_lib.sh` (by default it requires Python 3.6. If you are using other Python versions, modify the command inside `compile_pers_lib.sh`).
- Run `train.py` with the commands like below:
```
python train.py --every 5 --start_clean 30 --k_cc 4 --k_outlier 32 --seed 77 --type uniform --noise 0.4 --patience 65 --gpus 0 --dataset cifar10 --zeta 0.5
```
- For point cloud dataset, run the command with `pc` argument:
```
python train.py --gpus 2 --every 5 --start_clean 10 --k_outlier 30 --k_cc 100 --noise 0.8 --type uniform --patience 60 --seed 77 --dataset pc --net pc --milestone 35 --zeta 2
```
Here the major parameters are:
- `every`: the frequency of data collection.
- `start_clean`: when to start data collection.
- `k_cc`: the parameter for computing the KNN graph when finding the largest connected component.
- `k_outlier`: the parameter for computing the KNN graph when applying zeta filtering.
- `seed`: the random seed.
- `type`: the noise type. Options include `uniform` and `asym`.
- `noise`: the noise level.
- `patience`: this is a trick to save training time. If we observe no obvious improvement of validation accuracy for a consecutive number of `N` epochs, we stop the training.
- `gpus`: run on which GPU.
- `dataset`: which dataset to use. Options include `cifar10`, `cifar100` and `pc`. For the `pc` dataset, it can be downloaded from https://github.com/charlesq34/pointnet
- `zeta`: the parameter for `zeta` filtering. **Note that, when setting zeta to be > 1.0, we will use majority voting to remove the outliers. This sometimes achieves better performance.**

Practical tips: For the extrmely noisy scenarios (noise level >= 0.8), we observe setting a larger `k_cc` is better.

_Our code will be further improved to make it cleaner and easier to use._

## Reference:
```
@inproceedings{wu2020topological,
  title={A Topological Filter for Learning with Label Noise},
  author={Wu, Pengxiang and Zheng, Songzhu and Goswami, Mayank and Metaxas, Dimitris and Chen, Chao},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
## Related Works:

- Error-Bounded Correction of Noisy Labels. [[Paper]](https://arxiv.org/pdf/2011.10077.pdf)[[Code]](https://github.com/pingqingsheng/LRT)
- Learning with Feature Dependent Label Noise: A Progressive Approach. [[Paper]](https://openreview.net/pdf?id=ZPa2SyGcbwh)[[Code]](https://github.com/pxiangwu/PLC)
