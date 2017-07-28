cd /home/legion/CDVTL/SSD/caffe
./build/tools/caffe train \
--solver="models/VGGNet/SHOPPE_2017/SSD_300x300/solver.prototxt" \
--snapshot="models/VGGNet/SHOPPE_2017/SSD_300x300/VGG_SHOPPE_2017_SSD_300x300_iter_36915.solverstate" \
--gpu 0 2>&1 | tee jobs/VGGNet/SHOPPE_2017/SSD_300x300/VGG_SHOPPE_2017_SSD_300x300.log
