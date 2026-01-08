#!/bin/bash
python spatial_actor/train.py \
  --device 0,1,2,3,4,5,6,7 \
  --iter-based \
  --data-folder YOUR_DATA_FOLDER \
  --train-replay-dir YOUR_REPLAY_DIR \
  --log-dir YOUR_LOG_DIR \
  --cfg_path spatial_actor/configs/spact.yaml \
  --cfg_opts ""
