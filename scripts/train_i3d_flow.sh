
python train_i3d.py \
    -dataset 'uavhuman' \
    -mode 'flow' \
    -save_model 'D:/repos/i3d_crf/models/flow/' \
    -root_train 'D:/data/pose_detection_sample/kinectrgbs' \
    -train_split 'D:/data/pose_detection_sample/train_test_split.json' \
    -root_eval 'D:/data/pose_detection_sample/kinectrgbs' \
    -eval_split 'D:/data/pose_detection_sample/train_test_split.json' \
    -snippets 64 \
    -batch_size 1 \
    -batch_size_eval 1 \
    -saving_steps 10 \
    -num_steps_per_update 1 \
    -num_classes 400 \
    -max_steps 100 \
    -init_lr 0.1 \
    -use_cls True \
    -num_workers 1

# original base model (i3d)
# python train_i3d.py \
#     -dataset 'charades' \
#     -mode 'flow' \
#     -save_model 'path_to_saving_directory' \
#     -root_train 'path_to_flow_training_data' \
#     -train_split 'path_to_train_charades.json' \
#     -root_eval 'path_to_flow_evaluation_data' \
#     -eval_split 'path_to_test_charades.json' \
#     -snippets 64 \
#     -batch_size 4 \
#     -batch_size_eval 4 \
#     -saving_steps 5000 \
#     -num_steps_per_update 1 \
#     -num_classes 157 \
#     -init_lr 0.1 \
#     -use_cls True

# original full crf
#     -crf True \
#     -conditional_crf True \
#     -reg_crf 1e-3