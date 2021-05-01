# Computer Vision
Author: Haitian Jiang, Yuankun Yang, Siyu Yuan

***Model parameters are stored in `output-xxx/xxx-param.pth`***



## Model Factors experiments
### Convolution Kernels
```
nohup python3 main-template.py --experiment_name channel_size_16 --init_channel_size 16 &>channel_size_16.out
nohup python3 main-template.py --experiment_name channel_size_32 --init_channel_size 32 &>channel_size_32.out
nohup python3 main-template.py --experiment_name channel_size_64 --init_channel_size 64 &>channel_size_64.out
```
### Batch Normalization
```
nohup python3 main-template.py --experiment_name no_BN2 --init_channel_size 64  --batch_size 64 --batch_normalization 0 &>no_BN2.out
nohup python3 main-template.py --experiment_name BN2 --init_channel_size 64  --batch_size 64 --batch_normalization 1 &>BN2.out
```
### Shortcut
```
nohup python3 main-template.py --experiment_name no_shortcut --shortcut 0 &>no_shortcut.out
nohup python3 main-template.py --experiment_name shortcut --shortcut 1 &>shortcut.out
```
## Training Process Factors
### Optimizer
```
nohup python3 main-template.py --experiment_name adam3 --init_channel_size 64  --batch_size 64 --optimizer 0 --epoch 64 &>adam3.out(origion adam cannot train)
nohup python3 main-template.py --experiment_name SGD3 --init_channel_size 64  --batch_size 64 --optimizer 1 --epoch 64 &>SGD3.out
```
### Activation Function
```
nohup python3 main-template.py --experiment_name RELU &>RELU.out
nohup python3 main-template.py --experiment_name Sigmoid &>Sigmoid.out
nohup python3 main-template.py --experiment_name Tanh &>Tanh.out
nohup python3 main-template.py --experiment_name LEAKYRELU &>LEAKYRELU.out
nohup python3 main-template.py --experiment_name ELU&>ELU.out
```
### Loss Function
```
nohup python3 main-template.py --experiment_name CE_loss --loss_function 0 &>CE_loss.out
nohup python3 main-template.py --experiment_name MSE_loss --loss_function 1 &>MSE_loss.out
nohup python3 main-template.py --experiment_name L1Loss --loss_function 2 &>L1Loss.out
```
### Learning Rate
```
nohup python3 main-template.py --experiment_name OnPlateau --scheduler 1 &>OnPlateau.out
nohup python3 main-template.py --experiment_name ExponentialLR --scheduler 3 &>ExponentialLR.out
nohup python3 main-template.py --experiment_name CosineAnnealingLR --scheduler 4 &>CosineAnnealingLR.out
```
### Batch Size
```
nohup python3 main-template.py --experiment_name batch_16 --batch_size 16 &>batch_16.out
nohup python3 main-template.py --experiment_name batch_32 --batch_size 32 &>batch_32.out
nohup python3 main-template.py --experiment_name batch_64 --batch_size 64 &>batch_64.out
```
### Weight Decay
```
nohup python3 main-template.py --experiment_name wd_5e-4 --weight_decay 0.0005 &>wd_5e-4.out
nohup python3 main-template.py --experiment_name wd_1e-3 --weight_decay 0.001 &>wd_1e-3.out
nohup python3 main-template.py --experiment_name wd_1e-4 --weight_decay 0.0001 &>wd_1e-4.out
```



