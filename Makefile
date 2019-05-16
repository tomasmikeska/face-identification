
DATASET=data/dataset.npy

setup-env:
	pip install -r requirements.txt

compress-dataset:
	python src/dataset.py

floyd-train:
	floyd run --task train

floyd-train-cpu:
	floyd run --task train-cpu

local-train:
	python src/train.py
