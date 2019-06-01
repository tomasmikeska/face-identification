
DATASET=data/dataset.npy

setup-env:
	pip install -r requirements.txt

compress-dataset:
	python src/dataset.py

floyd-process-dataset:
	floyd run --task process_dataset

floyd-train:
	floyd run --task train

local-train:
	python src/train.py
