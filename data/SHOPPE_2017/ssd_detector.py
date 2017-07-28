'''
Created on 9 Jan 2017

@author: mozat
'''
import cv2
import sys
sys.path.insert(1,'/home/legion/auto_tag/ssd_deja_autotag/caffe/python')
sys.path.insert(1,'/home/legion/CDVTL/SSD/caffe/python')
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os
import glob
from matplotlib import pyplot as plt
#sys.path.insert(1,'/home/legion/auto_tag/auto_tag/src')
import Get_image

img_base_url = 'http://office.mozat.com/photos/product_images/'
img_dir = '/media/Algorithm/product_images/'

class SSDClothDetector(object):
    VISUALIZE_THRESHOLD = 0.1
    def __init__(self, caffe_net_model, caffe_net_path,label_net_path):
        self.caffe_net_model = caffe_net_model
        self.caffe_net_path = caffe_net_path
        self.label_net_path = label_net_path
        self.label_map = self._get_label()
        self._ssd_init()
        self.get_img = Get_image.get_img
        super(SSDClothDetector, self).__init__()
        
    def _get_label(self):
        if hasattr(self, 'labelmap') and self.labelmap:
            return self.labelmap
        fp = open(self.label_net_path, 'r')
        labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(fp.read()), labelmap)
        num_labels = len(labelmap.item)
        label_map = {}
        for idx in xrange(0, num_labels):
            label_map[labelmap.item[idx].label] = labelmap.item[idx].display_name
        #label_map = {0: u'background', 1: u'hat', 2: u'glasses', 3: u'top', 4: u'shorts', 5: u'skirt', 6: u'trousers', 7: u'bag', 8: u'shoes', 9: u'dress', 10: u'outwear', 11: u'one_piece'}
        return label_map
    
    def _ssd_init(self):
        caffe.set_mode_gpu() #3 images using 1 seconds, using multithreading it wil bump out initilization error
        #caffe.set_device(0)
        #caffe.set_mode_cpu() #3 images using 6 to 10 seconds with 4 threads

        caffe_mean_values = np.array([104, 117, 123])
        self.net = caffe.Net(self.caffe_net_path, self.caffe_net_model, caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_mean('data', caffe_mean_values)
        
    def detect(self, img_path):
        try:
            img = self.get_img(img_path)
        except:
            print "image path error"
            return None
        
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img)
        out = self.net.forward()
        rs = out['detection_out']
        confidences = rs[:, :, :, 2]
        confidences = np.reshape(confidences, (confidences.size, 1))        
        detection_out = []
        for v in range(0, len(confidences)):
            record = rs[:, :, v, :]
            label = int(record[:, :, 1])
            score = float(record[:, :, 2])
            if score<self.VISUALIZE_THRESHOLD: #filter out bad result
                continue
            xmin = int(record[:, :, 3] * img.shape[1])
            ymin = int(record[:, :, 4] * img.shape[0])
            xmax = int(record[:, :, 5] * img.shape[1])
            ymax = int(record[:, :, 6] * img.shape[0])
            result = dict([('score', round(score,3)), ('bbox', [float(xmin), float(ymin), float(xmax), float(ymax)]),
                          ('label', label),('label_name', self.label_map[label])])
            detection_out.append(result)
        return detection_out
    
    def detect_post_processing(self, img_path):
        detection_out = self.detect(img_path)
        if not detection_out:
            return None
        
        postResultsDict = {}
        for item in detection_out:
            if item['score'] < 0.3:
                continue
            label = item['label']
            if label in postResultsDict and postResultsDict[label]['score'] < item['score']:
                    postResultsDict[label] = item
            else:
                postResultsDict[label] = item 
        postResults = list(postResultsDict.values())
        return postResults
        
    def showResults(self, img_file, results, outfile_name, labelmap=None, threshold=None, display=None):
        if not os.path.exists(img_file):
            print "{} does not exist".format(img_file)
            return
        if results is None:
            print "{} empty detection".format(img_file)
            return
        img = io.imread(img_file)
        plt.clf()
        plt.imshow(img)
        plt.axis('off');
        ax = plt.gca()
        num_classes = len(labelmap.item) if labelmap else 59
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
        for res in results:
            if 'score' in res and threshold and float(res["score"]) < threshold:
                continue
            name = self.label_map[res['label']]
            color = colors[res['label'] % num_classes]
            bbox = res['bbox']
            coords = (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1]
            ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=3))
            if 'score' in res:
                score = res['score']
                display_text = '%s: %.2f' % (name, score)
            else:
                display_text = name
            ax.text(bbox[0], bbox[1], display_text, bbox={'facecolor':color, 'alpha':0.5})

        #plt.show()
        plt.savefig(outfile_name+'.jpg', bbox_inches="tight")
        os.rename(outfile_name+'.jpg',outfile_name)
    
if __name__ == '__main__':
    import os
    #caffe_root = os.getcwd().split('examples')[0]
    ## simple SSD with 12 labels
    # caffe_root = '/home/legion/auto_tag/ssd_deja_autotag/caffe'
    # caffe_net_path = os.path.join(caffe_root, 'models/VGGNet/cloth_detect/SSD_300x300_score/deploy.prototxt')
    # caffe_net_model = os.path.join(caffe_root, 'models/VGGNet/cloth_detect/SSD_300x300/VGG_cloth_detect_SSD_300x300_iter_17000.caffemodel')
    # caffe_label_path = os.path.join(caffe_root, 'data/cloth_detect/labelmap_cloth.prototxt')
    # detector = SSDClothDetector(caffe_net_model, caffe_net_path, caffe_label_path# )
    # img_paths = glob.glob(os.path.join(caffe_root, 'examples/ssd/test_imgs/*.jpg'))
    
    caffe_root = '/home/legion/CDVTL/SSD/caffe'
    caffe_net_path = os.path.join(caffe_root, 'models/VGGNet/SHOPPE_2017/SSD_300x300/deploy1.prototxt')
    caffe_net_model = os.path.join(caffe_root, 'models/VGGNet/SHOPPE_2017/SSD_300x300/VGG_SHOPPE_2017_SSD_300x300_iter_28080.caffemodel')
    caffe_label_path = os.path.join(caffe_root, 'data/SHOPPE_2017/labelmap_voc.prototxt')
    detector = SSDClothDetector(caffe_net_model, caffe_net_path, caffe_label_path)

    img_paths = glob.glob(os.path.join(caffe_root, 'data/SHOPPE_2017/JPEGImages/test/*.jpg'))

    for (idx,img_path) in enumerate(img_paths[:1000]):
        result = detector.detect_post_processing(img_path)
        print result
        destpath = os.path.join(caffe_root, 'data/SHOPPE_2017/results/detections', os.path.basename(img_path)[:-4]+'_detection.jpg')
        detector.showResults(img_path, result, destpath)
    pass
    


