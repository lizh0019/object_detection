cd /home/legion/CDVTL/SSD/caffe
./build/tools/caffe train \
--solver="models/VGGNet/WIDERFACE_2016/SSD_300x300/solver.prototxt" \
--snapshot="models/VGGNet/WIDERFACE_2016/SSD_300x300/VGG_WIDERFACE_2016_SSD_300x300_iter_3520.solverstate" \
--gpu 0 2>&1 | tee jobs/VGGNet/WIDERFACE_2016/SSD_300x300/VGG_WIDERFACE_2016_SSD_300x300.log
