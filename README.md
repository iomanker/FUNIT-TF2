# FUNIT-TF2: Few-Shot Unsupervised Image-to-Image Translation in Tensorflow 2
![IntroImage](https://github.com/iomanker/FUNIT-TF2/blob/master/public/intro_image.png)
## Update
* (2021/05/12) Found Tensorflow 2.3 has Distributed Training problem, update to Tensorflow 2.4.0
* (2021/01/11) Here is a big changed update, and making more pythonic. This also supports Tensorflow 2.3 now.
## Installation
* Clone: `git clone https://github.com/iomanker/FUNIT-TF2.git`
* Install CUDA11.0+, cuDNN8.0+
* Install required python pakcages
    * `pip install tensorflow-gpu==2.4.0`
    * `pip install matplotlib`
    * `pip install pyyaml`

## Dataset Preparation
This step is followed by original FUNIT. Please click [here](https://github.com/NVlabs/FUNIT/#dataset-preparation).
## Training
### Arguments 
| Args              | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `config`          | a path of config yaml file                                   |
| `output_path`     | a path of results of images' output                          |
| `ckpt_path`       | a path of saved checkpoints                                  |
| `log_path`        | a path of TensorBoard Event                                  |
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
* `test.py`: to show model's inference image.
* `datasets.py`: Processing raw data into tf.data.Dataset, You should pay more attention on it.
* `losses.py`: All of loss functions are here.
* `containers.py`: `FUNIT` model, `Generator`, `Discriminator`.
* `models.py`: `Encoder`, `Decoder` etc.
* `blocks.py`: Blocks of `Conv2D`, `ResIdentity`, etc.
* `layers.py`: `InstanceNorm`, `AdaIN`, `ReflectionPadding`.
* `utils.py`: Some useful functions. 