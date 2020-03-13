# FUNIT-TF2: Few-Shot Unsupervised Image-to-Image Translation in Tensorflow 2
![IntroImage](https://github.com/iomanker/FUNIT-TF2/blob/master/public/intro_image.png)
## Installation
* Clone: `git clone https://github.com/iomanker/FUNIT-TF2.git`
* Install CUDA10.0+, cuDNN7.5
* Install required python pakcages
    * `pip install tensorflow-gpu`
    * `pip install matplotlib`
    * `pip install pyyaml`
## Hardware & Computation Comparison
| List            | FUNIT                       | FUNIT-TF2               |
| --------------- | --------------------------- | ----------------------- |
| Hardware        | **4** Nvidia V100 32GB GPUs | Same as the former      |
| Workstation     | Acer-AiForge x NCTU         | Same as the former      |
| Library Version | Pytorch 1.4.0               | Tensorflow 2.0          |
| Dataset         | ImageNet Animal Dataset     | ImageNet Animal Dataset |
| Iteration Time  | 4.2+ s                      | 1.0+ s                  |

## Dataset Preparation
This step is followed by original FUNIT. Please click [here](https://github.com/NVlabs/FUNIT/#dataset-preparation).
## Training
### Arguments 
| Args              | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `config`          | a path of config yaml file                                   |
| `output_path`     | a path of results of images' output                          |
| `ckpt_path`       | a path of saved checkpoints                                  |
| `multigpus`       | Whether or not turn on multi-gpus                            |
| `test_batch_size` | a number of produced test images                             |
| `resume`          | Whether or not continue training by former stored checkpoint |
### Command
```
python train.py --config configs/funit_animals.yaml --multigpus
```
## Introduction of Files
### Main
* `train.py`: a main entry to train network.
* `run_step.py`: `train_step` & `test_step` on sigle GPU.
* `datasets.py`: Processing raw data into tf.data.Dataset.
* `losses.py`: All of loss functions are here.
* `containers.py`: `FUNIT` model, `Generator`, `Discriminator`.
* `models.py`: `Encoder`, `Decoder` etc.
* `blocks.py`: Blocks of `Conv2D`, `ResIdentity`, etc.
* `layers.py`: `InstanceNorm`, `AdaIN`, `ReflectionPadding`.
* `utils.py`: Some used functions. 
### Other
* `train_prototype.py`: a test file to train while using limited single GPU.