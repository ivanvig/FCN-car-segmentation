# FCN car segmentator

A car segmentation fully convolutional network

## Getting Started

### Prerequisites

In order to execute this code the following python libraries are needed:

* tensorflow
* ...
* otros

#### Easy way
Install the necessary packages to get the dataset:
```
$ apt install wget unzip
```

Then, inside de repository folder execute
```
$ data/get_data.sh
```

#### Manual way
Alternatively, you can download the dataset from [here](https://lear.inrialpes.fr/people/marszalek/data/ig02/ig02-v1.0-cars.zip), and decompress it in data/ig02-cars.

## Running the network

The code is distributed in four files:

* [fcn.py](src/fcn.py)
* [fcn_train.py](src/fcn_train.py)
* [fcn_eval.py](src/fcn_eval.py)
* [fcn_input.py](src/fcn_input.py)

### fcn.py

The network architecture is implemented in this file. If you want to change something related to the architecture (layers, loss function, etc) this is the place.

### fcn_input.py

The dataset preprocessing happens here, to modify anything related with the input to the network, you can do it here.

### fcn_train.py

This is the script that must be executed to train the network.
```
$ python fcn_train.py
```
Attention: This script will destroy previous training checkpoints

If you want to watch the progress, run
```
$ tensorboard --logdir=/tmp/ig02_train
```

You can change the logdir modifing a line in [fcn_train.py](src/fcn_train.py), remember to modify accordingly [fcn_eval.py](src/fcn_eval.py) as well.


### fcn_eval.py

This is the script that evaluates the network being trained using the evaluation set.

```
python fcn_eval.py
```
Attention: This script will destroy previous evaluating checkpoints

If you want to watch the progress, run
```
$ tensorboard --logdir=/tmp/ig02_train
```

## References

* [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
* [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
* [Convolutional Neural Network - Tensorflow](https://www.tensorflow.org/tutorials/deep_cnn)
