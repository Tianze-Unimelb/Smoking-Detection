# Smoking-detection

## 1.Introduction

Image classification pipeline based on PyTorch, the training framework uses `Pytorch-Base-Trainer (PBT)`.
**Due to github rules, files larger than 25MB cannot be uploaded. For convenience, please download the complete code here https://pan.baidu.com/s/1mNB1IH4Rv-sVrb47Pt47rg PIN:6666**

- Github: Pytorch-Base-Trainer: [Pytorch distributed training framework](https://github.com/PanJinquan/Pytorch-Base-Trainer)
- pip install package： [basetrainer](https://pypi.org/project/basetrainer/)
- Pytorch basic training library: Pytorch-Base-Trainer(Support model pruning and distributed training)[Tutorial](https://panjinquan.blog.csdn.net/article/details/122662902)


## 2.安装
- Depends on Python packages: [requirements.txt](./requirements.txt)

```bash
# Install Anaconda3 first
# Create a new virtual environment pytorch-py36 in conda (if it already exists, no need to create a new one)
conda create -n pytorch-py36 python==3.6.7
# Activate the virtual environment pytorch-py36 (need to run every time)
conda activate pytorch-py36
# Install the project dependent packages (if already installed, no need to install)
pip install -r requirements.txt
```

## 3.Data: Prepare Train and Test data

- Train and Test datasets require images of the same category to be placed in the same folder; and the sub-directory folders are named after the category name

![](docs/98eb1599.png)

- Category file: one list per line: [class_name.txt](data/dataset/class_name.txt) (please press Enter for the last line)

![](docs/37081789.png)

- Modify the configuration file data path: [config.yaml](configs/config.yaml)
```yaml
train_data: # Multiple datasets can be added
- 'data/dataset/train'
- 'data/dataset/train2'
test_data: 'data/dataset/test'
class_name: 'data/dataset/class_name.txt'
```

## 4.Train
```bash
python train.py -c configs/config.yaml 
```

- The target supports backbones such as googlenet, inception_v3, resnet[18,34,50], mobilenet_v2, etc. For details, see [backbone](classifier/models/build_models.py), etc.
Other backbones can be added by customization
- Training parameters can be configured in the [config.yaml](configs/config.yaml) configuration file

| **Parameter**      | **type**      | **Reference value**   | **Illustrate**                                       |
|:-------------|:------------|:------------|:---------------------------------------------|
| train_data   | str, list   | -           | Training data file, can support multiple files                               |
| test_data    | str, list   | -           | Test data file, can support multiple files                              |
| class_name   | str         | -           | Category File                               |
| work_dir     | str         | work_space  | Training Output Workspace                                    |
| net_type     | str         | resnet18    | Backbone Type,{resnet18/34/50,mobilenet_v2,googlenet,inception_v3} |
| input_size   | list        | [128,128]   | Model input size[W,H]                                  |
| batch_size   | int         | 32          | batch size                                   |
| lr           | float       | 0.1         | Initial learning rate size                                     |
| optim_type   | str         | SGD         | Optimizer，{SGD,Adam}                               |
| loss_type    | str         | CELoss      | Loss Function                                         |
| scheduler    | str         | multi-step  | Learning rate adjustment strategy，{multi-step,cosine}                  |
| milestones   | list        | [30,80,100] | Nodes that reduce learning rate，Only valid when scheduler=multi-step            |
| momentum     | float       | 0.9         | SGD Momentum Factor                                      |
| num_epochs   | int         | 120         | Number of training cycles                                      |
| num_warn_up  | int         | 3           | Number of warn_up                                   |
| num_workers  | int         | 12          | Number of DataLoader threads started                              |
| weight_decay | float       | 5e-4        | Weight decay coefficient                                       |
| gpu_id       | list        | [ 0 ]       | Specify the GPU card number for training. You can specify multiple                             |
| log_freq     | in          | 20          | The frequency of displaying LOG information                                   |
| finetune     | str         | model.pth   | Finetune the model                                 |
| progress     | bool        | True        | Whether to display the progress bar                                      |
| distributed  | bool        | False       | Whether to use distributed training                                    |

## 5.Test Demo

- First modify[demo.py](demo.py)

```python configuration file
def get_parser():
    # Configuration file
    config_file = "configs/config.yaml"
    # Model file
    model_file = "work_space/mobilenet_v2/model/latest_model_099_97.5248.pth"
    # Image directory to be tested
    image_dir = "data/test_image"
    parser = argparse.ArgumentParser(description="Inference Argument")
    parser.add_argument("-c", "--config_file", help="configs file", default=config_file, type=str)
    parser.add_argument("-m", "--model_file", help="model_file", default=model_file, type=str)
    parser.add_argument("--device", help="cuda device id", default="cuda:0", type=str)
    parser.add_argument("--image_dir", help="image file or directory", default=image_dir, type=str)
    return parser
```

- Then run demo.py

```bash
python demo.py
```

## 6.Visualization

Currently, the training process visualization tool is Tensorboard. How to use it:

```bash
tensorboard --logdir=path/to/log/
```

## 7.Other
