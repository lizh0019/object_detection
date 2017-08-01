# Caffe Documentation

0) Environments: Caffe, python, GPU, Ubuntu.
   Dataset: https://github.com/bearpaw/clothing-co-parsing, as it has 1003 images (1 annotation is wrong out of 1004) with pixel-level annotation of 59 object types.
   Detector: https://github.com/weiliu89/caffe/tree/ssd#preparation, as it is state-of-the-art detector that is claimed to be comparable to faster-RCNN, and the code is easy to configurate and modify.
   NOTE: All code are tested, however, you may need to change paths to your own username/path when any code cannot run.
   
1) Run SSD/caffe/data/SHOPPE_2017/show_pixel_anno.m in MATLAB, it will read pixel-level annotations stored in ".mat" files of the coparsing dataset and export bounding boxes to coparser_train_bbx_gt.txt and coparser_val_bbx_gt.txt, as well as other image information to trainval.txt, test.txt and test_name_size.txt

2) python SSD/caffe/data/SHOPPE_2017/create_list.py, it will convert the bounding box text files to xml annotation files for each image.

3) Run SSD/caffe/data/SHOPPE_2017/create_data.sh in shell command, it will convert the training and validation images to lmdb format.

4) python SSD/caffe/examples/ssd/ssd_pascal.py, it will automatically generate Caffe prototxt files and solver, and start training the VGG model using Single Shot Detector (SSD).

5) The models are automatically saved in the folder SSD/caffe/models/VGGNet/SHOPPE_2017/SSD_300x300/. Manually change the relative paths to absolute paths in the file SSD/caffe/models/VGGNet/SHOPPE_2017/SSD_300x300/deploy.prototxt and save as SSD/caffe/models/VGGNet/SHOPPE_2017/SSD_300x300/deploy1.prototxt

6) python SSD/caffe/data/SHOPPE_2017/ssd_detector.py, it will evaluate the 100K test images and save in the folder /media/legion/Work/SSD/caffe/data/SHOPPE_2017/results/detections. 

7) Performance: 
SSD is good for large object detection, but bad for detecting small objects.
for 59 original labels from coparser dataset, and export separate bounding boxes when an object is disconnected by other objects, the training process appears overfitting, and Mean Average Precision (MAP) is around 0.22;
When merging similar labels, e.g. merge earings and belts as accessaries, we get 13 labels, and export separate bounding boxes when an object is disconnected by other objects, MAP is around 0.37;
When using the merged 13 labels, and export only 1 bounding box when an object is disconnected by other objects, MAP is around 0.5;
When using fewer main labels (exclude accessories, hair, skin, socks, bra, etc., and keep tops, outerwear, pants, skirts, dress only), MAP is around 0.8;
In this experiment, we only show the detection results with 59 original labels. The detection results on 209 additional test images are available in SSD/caffe/data/SHOPPE_2017/results/detections_209.tar.gz, you can change path to generate the detection results on 100K test images.



