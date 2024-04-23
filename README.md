# VSGMN
### Requirements
The code implementation of **VSGMN** mainly based on [PyTorch](https://pytorch.org/). All of our experiments run and test in Python 3.8.8. To install all required dependencies:
```
$ pip install -r requirements.txt
```

## Training

We trained the model on three popular ZSL benchmarks: [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html) and [AWA2](http://cvml.ist.ac.at/AwA2/) following the data split of [xlsa17](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip). 
Please follow [TransZero](https://github.com/shiming-chen/TransZero) to prepare datasets and extract visual features.

### Training Script

```
$ python train_awa_vsgmn.py
$ python train_cub_vsgmn.py
$ python train_sun_vsgmn.py
