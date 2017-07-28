import os,sys
import glob
import cv2

base_dir = '/home/legion/CDVTL/SSD/caffe'

train_image_paths = os.path.join(base_dir, 'data/SHOPPE_2017/JPEGImages/trainval/*.jpg')
train_image_paths = glob.glob(train_image_paths)

test_image_paths = os.path.join(base_dir, 'data/SHOPPE_2017/JPEGImages/test/*.jpg')
test_image_paths = glob.glob(test_image_paths)

Tag_list = ['none_of_the_above','accessories','bag','belt','blazer','blouse','bodysuit','boots','bra','bracelet','cape','cardigan','clogs','coat','dress','earrings','flats','glasses','gloves','hair','hat','heels','hoodie','intimate','jacket','jeans','jumper','leggings','loafers','necklace','panties','pants','pumps','purse','ring','romper','sandals','scarf','shirt','shoes','shorts','skin','skirt','sneakers','socks','stockings','suit','sunglasses','sweater','sweatshirt','swimwear','t-shirt','tie','tights','top','vest','wallet','watch','wedges']

def import_bbox(bbox_filename,dataset):
    BBoxes = {}
    with open(bbox_filename,'r') as bbox_file:
        count = 0
        for line in bbox_file:
            if count == 0:
                if line[-4:-1] == 'jpg':
                    key = ('data/SHOPPE_2017/JPEGImages/' + dataset + line)[:-1]
                    BBoxes[key] = {}
                    count = -2
                else:
                    tag_id = int(line)
                    BBoxes[key][tag_id]=[]
                    count = -1
            elif count == -2:
                tag_id = int(line)
                BBoxes[key][tag_id]=[]
                count = -1
            elif count == -1:
                count = int(line)
            else:
                bbox = map(lambda r: int(r), line.split()[:4])
                BBoxes[key][tag_id].append(bbox)
                count -= 1
            pass
    return BBoxes

def export_xml(BBoxes, xml_filename, img_name, img_shape):
    with open(xml_filename,'w') as xml_file:
        xml_file.write('<annotation>\n')
        xml_file.write('\t<folder>SHOPPE2017</folder>\n')
        xml_file.write('\t<filename>'+img_name+'</filename>\n')
        xml_file.write('\t<source>\n')
        xml_file.write('\t\t<database>The SHOPPE2017 Database</database>\n')
        xml_file.write('\t</source>\n')
        xml_file.write('\t<size>\n')
        xml_file.write('\t\t<width>'+str(img_shape[1])+'</width>\n')
        xml_file.write('\t\t<height>'+str(img_shape[0])+'</height>\n')
        xml_file.write('\t\t<depth>'+str(img_shape[2])+'</depth>\n')
        xml_file.write('\t</size>\n')
        xml_file.write('\t<segmented>0</segmented>\n')
        for tag_id,BBox in BBoxes.items():
            for bbox in BBox:
                if bbox[0]>=img_shape[1] or bbox[0]<0 or \
                   bbox[1]>=img_shape[0] or bbox[1]<0 or \
                   bbox[2]<=0 or bbox[3]<=0 or \
                   bbox[0]+bbox[2]>=img_shape[1] or \
                   bbox[1]+bbox[3]>=img_shape[0]: #weired bbox for face
                    continue

                xml_file.write('\t<object>\n')
                xml_file.write('\t\t<name>'+Tag_list[tag_id]+'</name>\n')
                xml_file.write('\t\t<pose>Unspecified</pose>\n')
                xml_file.write('\t\t<truncated>0</truncated>\n')
                xml_file.write('\t\t<difficult>0</difficult>\n')
                xml_file.write('\t\t<bndbox>\n')
                xml_file.write('\t\t\t<xmin>'+str(bbox[0])+'</xmin>\n')
                xml_file.write('\t\t\t<ymin>'+str(bbox[1])+'</ymin>\n')
                xml_file.write('\t\t\t<xmax>'+str(min(img_shape[1]-1,bbox[0]+bbox[2]))+'</xmax>\n')
                xml_file.write('\t\t\t<ymax>'+str(min(img_shape[0]-1,bbox[1]+bbox[3]))+'</ymax>\n')
                xml_file.write('\t\t</bndbox>\n')
                xml_file.write('\t</object>\n')
                
        xml_file.write('</annotation>')

train_bboxes_filename = os.path.join(base_dir,'data/SHOPPE_2017/coparser_train_bbx_gt.txt')
train_bboxes = import_bbox(train_bboxes_filename, 'trainval/')

        
out_filename = 'trainval.txt'
with open(out_filename, 'w') as outf:
    for img_path in train_image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        leftname = '/'.join(img_path.split('/')[-5:])
        rightname = 'data/SHOPPE_2017/Annotations/'+os.path.basename(img_path).split('.')[0]+'.xml'
        #xml_filename = os.path.join(base_dir, 'data/SHOPPE_2017', rightname)
        xml_filename = os.path.join(base_dir, rightname)
        if not train_bboxes.get(leftname):
            continue
        export_xml(train_bboxes[leftname], xml_filename, os.path.basename(img_path),img.shape)
        line = leftname + ' ' + rightname
        outf.write(line)
        outf.write('\n')

test_bboxes_filename = os.path.join(base_dir,'data/SHOPPE_2017/coparser_val_bbx_gt.txt')
test_bboxes = import_bbox(test_bboxes_filename,'test/')

out_filename = 'test.txt'
with open(out_filename, 'w') as outf:
    for img_path in test_image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        leftname = '/'.join(img_path.split('/')[-5:])
        rightname = 'data/SHOPPE_2017/Annotations/'+os.path.basename(img_path).split('.')[0]+'.xml'
        #xml_filename = os.path.join(base_dir, 'data/SHOPPE_2017', rightname)
        xml_filename = os.path.join(base_dir, rightname)
        if not test_bboxes.get(leftname):
            continue
        export_xml(test_bboxes[leftname], xml_filename, os.path.basename(img_path),img.shape)
        line = leftname + ' ' + rightname
        outf.write(line)
        outf.write('\n')

out_filename = 'test_name_size.txt'
with open(out_filename, 'w') as outf:
    for img_path in train_image_paths+test_image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        line = os.path.basename(img_path).split('.')[-2] + ' ' + str(img.shape[0]) + ' ' + str(img.shape[1])
        outf.write(line)
        outf.write('\n')

'''
<annotation>
	<folder>VOC2012</folder>
	<filename>2007_000027.jpg</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
	</source>
	<size>
		<width>486</width>
		<height>500</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>boat</name>
		<pose>Left</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>19</xmin>
			<ymin>64</ymin>
			<xmax>424</xmax>
			<ymax>190</ymax>
		</bndbox>
	</object>
	<object>
		<name>person</name>
		<pose>Frontal</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>305</xmin>
			<ymin>155</ymin>
			<xmax>390</xmax>
			<ymax>375</ymax>
		</bndbox>
	</object>
</annotation>

'''
