# Face Identification (in-progress)

This project trains a SqueezeNet based FaceNet neural net to identify faces.

#### Requirements

- Python 3.x
- pip

### Installation and setup

Install pip packages using
```
$ pip install -r requirements.txt
```

Then put your local dataset into `data` file in project root. Your dataset should be split into 2 subfolders, `train` and `test`. Train and test folders should have a subfolder for each identity with his/hers id as folder name containing `{whatever}.png` files with recordings. Eval folder is flat and should contain png files directly.

Example:
```
.
+-- data
|   +-- train
|       +-- person1
|           ├── a1.png
|           └── a2.png
|       +-- person1
|           ├── b1.png
|           └── b2.png
|       +-- ...
|   +-- test
|       +-- person1
|           ├── a1.png
|           └── a2.png
|       +-- person1
|           ├── b1.png
|           └── b2.png
|       +-- ...
```

#### Usage

Train model using command (trained on LFW dataset from sklearn)
```
$ python src/facenet_train.py
```

After training the nn params are saved to `model/` folder.

Evaluate using command

```
$ python src/facenet_eval.py
```

🎉 🎉
