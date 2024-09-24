
# install
python=3.7
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch

## Start Up
Experiments are conducted on [AwA2](https://cvml.ist.ac.at/AwA2/), [CUB](http://www.vision.caltech.edu/visipedia/CUB-200.html)

We use AwA2 file format as default detailed in `./data/` folder and images should be downloaded and renamed as `./data/*/JPEGImages`. It is important to note that several cusomization work should be done for SUN dataset to maintain the same file format.

### Train

Use `experiments/run_trainer.py` to train the network. Run `help` to view all the possible parameters. We provide several config files under `./configs/` folder. Example usage:

```
python experiments/run_trainer.py --cfg ./configs/hybrid/VGG19_AwA2_PS_C.yaml
```

### Test

Use `experiments/run_evaluator.py` to evaluate the network with self_adaptation and `experiments/run_evaluator_hybrid.py` with hybrid method.

python experiments/run_evaluator_hybrid.py.py --cfg ./configs/hybrid/VGG19_AwA2_PS_C.yaml
