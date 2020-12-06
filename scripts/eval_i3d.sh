
python eval_i3d.py \
    -dataset 'uavhuman' \
    -mode 'rgb' \
    -save_model 'D:/repos/i3d_crf/models/rgb_imagenet.pt' \
    -root_eval 'D:/data/pose_detection_sample/kinect' \
    -eval_split 'D:/data/pose_detection_sample/train_test_split_orig.json' \
    -snippets 64 \
    -batch_size_eval 1 \
    -num_classes 154 \
    -crf False \
    -eval_checkpoint -1 \
    -num_workers 1

# original evaluation one stream (RGB of Optical-flow)
# pretrained is 400, charades is 157, uavhuman is 154
# python eval_i3d.py \
#     -dataset 'charades' 
#     -mode 'rgb' \
#     -save_model 'path_to_saving_directory' \
#     -root_eval 'path_to_rgb_evaluation_data' \
#     -eval_split  'path_to_test_charades.json' \
#     -snippets 64 \
#     -batch_size_eval 1 \
#     -num_classes 157 \
#     -crf True \
#     -eval_checkpoint 750000