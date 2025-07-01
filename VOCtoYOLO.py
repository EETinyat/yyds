
import random
import xml.etree.ElementTree as ET
import os
import shutil  # 导入shutil模块

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def xml_to_yolo(xml_path, labels_path, classes):
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    for xml_file in os.listdir(xml_path):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(xml_path, xml_file))
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            with open(os.path.join(labels_path, xml_file.replace('.xml', '.txt')), 'w') as label_file:
                for obj in root.iter('object'):
                    cls = obj.find('name').text
                    if cls not in classes:
                        continue
                    cls_id = classes.index(cls)
                    xmlbox = obj.find('bndbox')
                    b = (
                        float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                        float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)
                    )
                    bb = convert((w, h), b)
                    label_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

# 定义类别
classes = ['fire']
xml_path = r'E:\zhonghesheji\shuju\forPP\VOCdata2/Annotations'  # VOC 数据集的 XML 路径
labels_path = r'E:\zhonghesheji\shuju\forPP\VOCdata2/labels'  # 转换后的标签文件存放路径

# 执行转换
xml_to_yolo(xml_path, labels_path, classes)

# 设置数据集的路径
voc_root = r'E:\zhonghesheji\shuju\forPP\VOCdata2'
jpeg_images_dir = os.path.join(voc_root, 'JPEGImages')

# 获取所有图像文件的文件名（不包括扩展名）
image_ids = [f.split('.')[0] for f in os.listdir(jpeg_images_dir) if f.endswith('.jpg')]

# 随机打乱图像ID列表
random.shuffle(image_ids)

# 设置训练集和验证集的比例
train_ratio = 0.8
num_train = int(len(image_ids) * train_ratio)

# 分割训练集和验证集的图像ID
train_ids = image_ids[:num_train]
val_ids = image_ids[num_train:]

# 创建 train.txt 和 val.txt 文件
with open(os.path.join('dataset', 'train.txt'), 'w') as f:
    for image_id in train_ids:
        f.write(image_id + '\n')

with open(os.path.join('dataset', 'val.txt'), 'w') as f:
    for image_id in val_ids:
        f.write(image_id + '\n')

# 创建目标目录
os.makedirs(os.path.join('dataset', 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join('dataset', 'train', 'labels'), exist_ok=True)
os.makedirs(os.path.join('dataset', 'val', 'images'), exist_ok=True)
os.makedirs(os.path.join('dataset', 'val', 'labels'), exist_ok=True)

# 移动训练集图像和标签
for image_id in train_ids:
    image_path = os.path.join(jpeg_images_dir, f'{image_id}.jpg')
    label_path = os.path.join(labels_path, f'{image_id}.txt')
    target_image_path = os.path.join('dataset', 'train', 'images', f'{image_id}.jpg')
    target_label_path = os.path.join('dataset', 'train', 'labels', f'{image_id}.txt')
    if os.path.exists(image_path) and os.path.exists(label_path):
        shutil.copy(image_path, target_image_path)
        shutil.copy(label_path, target_label_path)

# 移动验证集图像和标签
for image_id in val_ids:
    image_path = os.path.join(jpeg_images_dir, f'{image_id}.jpg')
    label_path = os.path.join(labels_path, f'{image_id}.txt')
    target_image_path = os.path.join('dataset', 'val', 'images', f'{image_id}.jpg')
    target_label_path = os.path.join('dataset', 'val', 'labels', f'{image_id}.txt')
    if os.path.exists(image_path) and os.path.exists(label_path):
        shutil.copy(image_path, target_image_path)
        shutil.copy(label_path, target_label_path)