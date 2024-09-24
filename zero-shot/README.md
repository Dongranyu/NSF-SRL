# 环境 zero
# install
python=3.7
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
# 修改的文件
/data/ydr2021/ydr/AttentionZSL/configs/default.py 59行
/data/ydr2021/ydr/AttentionZSL/trainer.py 增加了32，121，122损失，可以替换为nn.BCEWithLogitsLoss()修改了125
数据集
https://cs.brown.edu/~gmpatter/sunattributes.html
######################################################## AWA2 dataset
# 对应的train和test命令
# train：python experiments/run_trainer.py --cfg ./configs/self_adaptation/VGG19_AwA2_PS_C.yaml
# test： python experiments/run_evaluator.py --cfg ./configs/self_adaptation/VGG19_AwA2_PS_C.yaml

# train： python experiments/run_trainer.py --cfg ./configs/hybrid/VGG19_AwA2_PS_C.yaml
# test： python experiments/run_evaluator_hybrid.py.py --cfg ./configs/hybrid/VGG19_AwA2_PS_C.yaml
##################################################################### CUB
Resnet101
# train：python experiments/run_trainer.py --cfg ./configs/self_adaptation/ResNet101_CUB_PS_C.yaml 
# test： python experiments/run_evaluator.py --cfg ./configs/self_adaptation/ResNet101_CUB_PS_C.yaml
# 修改文件路径
mean_field-posterior.py new_dataset.py new_mln.py preprocess.py
#####################################################################
# train：python experiments/run_trainer.py --cfg ./configs/self_adaptation/ResNet101_AwA2_PS_C.yaml
# test： python experiments/run_evaluator.py --cfg ./configs/self_adaptation/ResNet101_AwA2_PS_C.yaml
GOOLNET
/data/ydr2021/promot_learning/AttentionZSL/configs/self_adaptation/GoogleNet_CUB_PS_C.yaml
# train：python experiments/run_trainer.py --cfg ./configs/self_adaptation/GoogleNet_CUB_PS_C.yaml
# test： python experiments/run_evaluator.py --cfg ./configs/self_adaptation/GoogleNet_CUB_PS_C.yaml

# train：python experiments/run_trainer.py --cfg ./configs/self_adaptation/GoogleNet_CUB_PS_G.yaml
# test： python experiments/run_evaluator.py --cfg ./configs/self_adaptation/GoogleNet_CUB_PS_G.yaml



# train：python experiments/run_trainer.py --cfg ./configs/self_adaptation/GoogleNet_AwA2_PS_C.yaml
# test： python experiments/run_evaluator.py --cfg ./configs/self_adaptation/GoogleNet_AwA2_PS_C.yaml

# 文件保存
1. save model file and test results of every test category for each model file (ckpt_epoch_1)-> /data/ydr2021/ydr/AttentionZSL/checkpoint
2. save mean prediction results for each model file (ckpt_epoch_1)

# 数据集
AwA2   PS划分属于V1版，其中, 训练图片=trans_train_files.txt + trainval_PS.txt，共30414。测试图片=trans_test_files.txt，共6908。

############################################################################
# Attribute Attention for Semantic Disambiguation in Zero-Shot Learning

This repository contains the public release of the Python implementation of

**Attribute Attention for Semantic Disambiguation in Zero-Shot Learning**

Yang Liu, Jishun Guo, Deng Cai, Xiaofei He.

![framework](demo/framework.png)

If you use this code or find this work useful for your research, please cite:

```
@inproceedings{Liu_2019_ICCV,
  title={Attribute Attention for Semantic Disambiguation in Zero-Shot Learning},
  author={Liu, Yang and Guo, Jishun and Cai, Deng and He, Xiaofei},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  month={Oct},
  year={2019}
}
```

## Performance

### cZSL

![czsl](demo/czsl.png)

### gZSL

![gzsl](demo/gzsl.png)

## Start Up

Implemented and tested on Ubuntu 16.04 with Python 3.6 and Pytorch 1.0.1. Experiments are conducted on [AwA2](https://cvml.ist.ac.at/AwA2/), [CUB](http://www.vision.caltech.edu/visipedia/CUB-200.html) and [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html) datasets.

We use AwA2 file format as default detailed in `./data/` folder and images should be downloaded and renamed as `./data/*/JPEGImages`. It is important to note that several cusomization work should be done for SUN dataset to maintain the same file format.

## Basic Usage


### Train

Use `experiments/run_trainer.py` to train the network. Run `help` to view all the possible parameters. We provide several config files under `./configs/` folder. Example usage:

```
python experiments/run_trainer.py --cfg ./configs/self_adaptation/VGG19_AwA2_PS_C.yaml
```

Feel free to download the reported [checkpoints](https://drive.google.com/open?id=1mRO54hifHr4UW7D5oQ6D7lWzI4K4b5m2).

### Test

Use `experiments/run_evaluator.py` to evaluate the network with self_adaptation and `experiments/run_evaluator_hybrid.py` with hybrid method.

python experiments/run_evaluator.py --cfg ./configs/self_adaptation/VGG19_AwA2_PS_C.yaml