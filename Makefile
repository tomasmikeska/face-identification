include .env
export

remote-train:
	git archive -o paperspace.zip $(shell git stash create)
	zip -u paperspace.zip .env
	gradient jobs create \
		--name "face identification train" \
		--machineType "P5000" \
		--container "tomikeska/ml-box" \
		--workspaceArchive paperspace.zip \
		--command "make train"

train:
	pip3 install -r requirements.txt
	VGG_DATASET=/storage/datasets/VGGFace2/ \
	CASIA_DATASET=/storage/datasets/CASIA-WebFace/ \
	BB_TRAIN=/storage/datasets/bb_landmark/loose_bb_train.csv \
	CASIA_BB=/storage/datasets/casia_landmark.csv \
	MODEL_SAVE_PATH=/artifacts/ \
	TB_LOGS=/artifacts/tb_logs/ \
	LFW_DATASET=/storage/datasets/lfw/ \
	LFW_BB=/storage/datasets/lfw_landmark.csv \
	LFW_PAIRS=/storage/datasets/lfw_pairs.txt \
	python3 src/train.py

local-train:
	python src/train.py

test:
	PYTHONPATH=src/ python -m pytest tests/
