# Guided Attention Generative Network for Quality Enhancement of Compressed Video

### Requirements and dependencies

- [Pytorch 0.4.0](https://pytorch.org/)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [FlowNet2-Pytorch](https://github.com/mengab/flownet2-pytorch)
- [PWC-Net](https://github.com/mengab/PWC-Net) (Code and model are already included in this repository)

The code was developed using Python 3.6 & PyTorch 0.4 & CUDA 8.0 on Ubuntu 16.04. There may be a problem related to software versions. To fix the problem, you may look at the implementation in GAGNet files and replace the syntax to match the new PyTorch environment.

### Installation
Download repository:

    git clone https://github.com/mengab/GAGNet.git

Compile BiFlowNet dependencies (Channelnorm layers, Correlation and Resample2d):

    ./install.sh


### Code
Currently, we release our research code for training and testing for two types of models with content loss and combined loss. They should produce the same results as the paper under LD and AI configurations, respectively.

### Testing
* our input_mask test data can be downloaded from this link!
```
https://drive.google.com/drive/folders/
```

* It would be very easy to understand the test function and test on your own data.
* An example of test usage is shown as follows:
```bash 
$ cd ./GAGNet-master
$ Demo.sh
```
**Note:** 
* Please make sure the dependencies are complied successfully. 
* We provide two types of testing for different coding configuration in the `Demo.sh` file, you can comment one mode to test another. 
* For each coding configuration, we offer two pretrained models for the model with content loss and combined loss, respectively. 
* You can easily test the performance of the pretrained model under different configurations by modifying the test parameters in the `Demo.sh` file.


### Training

Train a new model:
```bash
$ cd ./GAGNet-master
$ python train.py --GPUs “xxx”
```

* We have provided some optional parameters in the `option.py` file, and specified all the default parameters in `train.py`. 
* We recommend that you read our `train.py ` and `option.py ` files carefully before training the model. 
* If you want to train the network model in the ablation study, you can choose to disable the corresponding parameter in the `option.py ` file. 

### Contact
We are glad to hear if you have any suggestions and questions. 
Please send email to xmengab@connect.ust.hk
