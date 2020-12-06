
python eval_i3d_2_streams.py \
    -dataset 'uavhuman' \
    -save_model_rgb 'D:/repos/i3d_crf/models/rgb/' \
    -save_model_flow 'D:/repos/i3d_crf/models/flow/' \
    -root_eval_rgb 'D:/data/pose_detection_sample/kinect' \
    -root_eval_flow 'D:/data/pose_detection_sample/kinectflows' \
    -eval_split 'D:/data/pose_detection_sample/train_test_split.json' \
    -snippets 64 \
    -batch_size_eval 1 \
    -crf True \
    -num_classes 400 \
    -eval_checkpoint_rgb -1 \
    -eval_checkpoint_flow -1 \
    -num_worker 1

# original evaluation two stream
# python eval_i3d_2_streams.py \
#     -dataset 'charades' \
#     -save_model_rgb 'path_to_rgb_saving_directory' \
#     -save_model_flow 'path_to_flow_saving_directory' \
#     -root_eval_rgb 'path_to_rgb_test_data' \
#     -root_eval_flow 'path_to_flow_test_data' \
#     -eval_split 'path_to_test_charades.json' \
#     -snippets 64 \
#     -batch_size_eval 1 \
#     -crf True \
#     -num_classes 157 \
#     -eval_checkpoint_rgb 500000 \
#     -eval_checkpoint_flow 500000