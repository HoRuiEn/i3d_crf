
python train.py \
    --max_epoch 100 \
    --mode 'flow' \
    --exp 'D:/data/pose_detection_sample/experiments' \
    --save_every 1 \
    --dataset 'uavhuman' \
    --root_train 'D:/data/pose_detection_sample/kinectrgbs_with_missing' \
    --train_split 'D:/data/pose_detection_sample/train_test_split.json' \
    --num_frames 64 \
    --num_classes 155 \
    --lr 0.002 \
    --batch_size 1 \
    --num_workers 1 \
    --weight_decay 0.0000001 \
    --resume 'D:/data/pose_detection_sample/experiments/steps0000008.pt'
