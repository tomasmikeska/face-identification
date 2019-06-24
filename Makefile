TB_PORT=9090

paperspace-train:
	git archive -o paperspace.zip $(shell git stash create)
	gradient jobs create \
		--name "face identification train" \
		--machineType "P4000" \
		--container "tomikeska/ml-box" \
		--workspaceArchive paperspace.zip \
		--ports $(TB_PORT):$(TB_PORT) \
		--command "make train"

train:
	pip3 install -r requirements.txt
	tensorboard --logdir=/artifacts/tb_logs/ --port=$(TB_PORT) &
	VGG_DATASET=/storage/datasets/VGGFace2/ \
	BB_TRAIN=/storage/datasets/bb_landmark/loose_bb_train.csv \
	MODEL_SAVE_PATH=/artifacts/ \
	TB_LOGS=/artifacts/tb_logs/ \
	python3 src/train.py

local-train:
	python3 src/train.py
