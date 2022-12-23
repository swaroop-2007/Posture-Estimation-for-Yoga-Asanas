#!/bin/bash
cd ./inference

python infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml --output-dir output_directory --image-ext mp4 input_directory
    
cd ..

cd ./data
python prepare_data_2d_custom.py -i C:/Users/swapn/Documents/BE_project/BE_project/video_pose/VideoPose3D/inference/output_directory -o myvideos

cd ..

python run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject 
input_directory --viz-action custom --viz-camera 0 --viz-video 
C:/Users/swapn/Documents/BE_project/BE_project/video_pose/VideoPose3D/inference/input_directory/ --viz-output output.mp4 --viz-size 6
echo Done!