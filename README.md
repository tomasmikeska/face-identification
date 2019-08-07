# Face Identification

This is a Keras implementation of face identification model using *ArcFace* loss and *Center loss*.

#### Requirements

- Python 3.x
- pip

#### Installation and setup

Install pip packages using
```
$ pip install -r requirements.txt
```

Download **VGGFace2** dataset for training to `data/` folder [here](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
(images and bb_landmarks) or **CASIA-WebFace** dataset [here](https://drive.google.com/file/d/1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz/view),
**LFW** dataset for validation [here](http://vis-www.cs.umass.edu/lfw/) (funneled images and pairs.txt).

[optional]

There is a Docker image included that was used for training in cloud. You can build it from local Dockerfile with
```
docker build -t ml-box .
```
or get it from Docker Hub
```
docker pull tomikeska/ml-box
```

#### Usage

Train model using command
```
$ python src/train.py
```

After training the weights are saved to `model/` folder by default. These weights contain all training layers (2 inputs, 2 outputs - softmax and centerloss) so in order to convert them to production model (single input and output) run command
```
$ python src/convert_model.py --weights model/densenet121_arcface_weights.h5 --nclasses 5386 -o model/densenet121_arcface_prod.h5
```

Evaluate on LFW using command

```
$ python src/lfw_validate.py --model model/densenet121_arcface_prod.h5
```

#### Training

Training took \~30 hours on NVIDIA P5000 on subset of VGGFace2 (~1M images, ~5K identities) depending on chosen base model.
