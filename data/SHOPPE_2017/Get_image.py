import urllib
import cv2
import numpy as np
import pdb

def get_img(img_path, resize=False):
    img_base_url = 'http://office.mozat.com/photos/product_images/'
    img_dir = '/media/Algorithm/product_images/'
    img = None
    if img_path == 'null' or img_path is None:
        return img

    try:
        img = cv2.imread(img_dir+img_path, cv2.IMREAD_COLOR)
    except:
        #pdb.set_trace()
        pass
    if img is None:#not local path
        # try http local path
        url = img_base_url+img_path
        resp = urllib.urlopen(url)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if img is None: # try full http
            resp = urllib.urlopen(img_path)
            img = np.asarray(bytearray(resp.read()), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    if resize:
        img = cv2.resize(img, (300,300), interpolation=cv2.INTER_CUBIC)
        
    return img
