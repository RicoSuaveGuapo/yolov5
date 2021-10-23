import os
import cv2
import shutil
import numpy as np
import argparse
import random
import json


def check_parser(opt):
    assert opt.ori_dir is not None
    assert os.path.exists(opt.ori_dir), 'No such directory'
    assert 'image' in os.listdir(opt.ori_dir), 'No "image" directory'
    assert 'annotations' in os.listdir(opt.ori_dir), 'No "annotations" directory'
    assert os.path.exists(opt.output_dir), 'Not existing output directory'


def copy_file(file_paths: list, opt, task, target_type):
    assert target_type in ['image', 'json']
    file_names = [os.path.basename(file_path) for file_path in file_paths]
    if target_type == 'image':
        file_output_paths = [os.path.join(opt.img_output_dir, task, img_name) for img_name in file_names]
    elif target_type == 'json':
        file_output_paths = [os.path.join(opt.json_output_dir, task, json_name) for json_name in file_names]
    [shutil.copyfile(file_path, file_output_paths[i]) for i, file_path in enumerate(file_paths)]
    return file_output_paths[0]


def xyxy2nxywh(bbox, size):
    h, w = size
    assert bbox.shape[0] == 2, 'the bbox format should be ([x1, x2,...], [y1, y2,...])'
    xmin = min(bbox[0])
    xmax = max(bbox[0])
    ymin = min(bbox[1])
    ymax = max(bbox[1])

    c_x = (xmin + xmax) / 2. / w
    c_y = (ymin + ymax) / 2. / h
    width = float(xmax - xmin) / w
    height = float(ymax - ymin) / h
    return [c_x, c_y, width, height]


def json2yolo(json_paths: list, opt, task: str):
    json_names = [os.path.basename(json_path).replace('json', 'txt') for json_path in json_paths]
    json_output_paths = [os.path.join(opt.lab_output_dir, task, json_name) for json_name in json_names]
    for i, json_path in enumerate(json_paths):
        with open(json_path) as f:
            annotation = json.load(f)
        shape_list = annotation['shapes']
        image_h = int(annotation['imageHeight'])
        image_w = int(annotation['imageWidth'])
        size = (image_h, image_w)
        for shape in shape_list:
            points = shape['points']  # [[x1, y1], [x2, y2], [x3, y3], ...]
            poly = np.array(points).astype(np.int32)
            box = xyxy2nxywh(poly.T, size)
            with open(json_output_paths[i], 'a') as f:
                f.write(f'0 {box[0]} {box[1]} {box[2]} {box[3]}\n')  # only one class support
    return json_output_paths[0]


def create_dir(opt):
    img_output_dir = os.path.join(opt.output_dir, 'images')
    lab_output_dir = os.path.join(opt.output_dir, 'labels')
    json_output_dir = os.path.join(opt.output_dir, 'jsons')
    if not os.path.exists(img_output_dir):
        os.mkdir(img_output_dir)
        os.mkdir(lab_output_dir)
        os.mkdir(json_output_dir)
    else:
        shutil.rmtree(img_output_dir)
        shutil.rmtree(lab_output_dir)
        shutil.rmtree(json_output_dir)
        os.mkdir(img_output_dir)
        os.mkdir(lab_output_dir)
        os.mkdir(json_output_dir)
        [os.mkdir(os.path.join(img_output_dir, task)) for task in opt.tasks]
        [os.mkdir(os.path.join(lab_output_dir, task)) for task in opt.tasks]
        [os.mkdir(os.path.join(json_output_dir, task)) for task in opt.tasks]

    opt.img_output_dir = img_output_dir
    opt.lab_output_dir = lab_output_dir
    opt.json_output_dir = json_output_dir

    # for visualization
    if not os.path.exists(opt.check_dir):
        os.mkdir(opt.check_dir)
    else:
        shutil.rmtree(opt.check_dir)
        os.mkdir(opt.check_dir)


def split_tasks(paths: list, opt):
    random.seed(opt.seed)
    random.shuffle(paths)
    train_end_index = int(len(paths) * (1 - opt.val_ratio - opt.test_ratio))
    val_end_index = int(len(paths) * (1 - opt.test_ratio))

    train_paths = paths[:train_end_index]
    val_paths = paths[train_end_index:val_end_index]
    test_paths = paths[val_end_index:]
    paths.sort()  # keep the orginal order

    return train_paths, val_paths, test_paths


def preprocess_dataset(opt):
    image_dir = os.path.join(opt.ori_dir, 'image')
    json_dir = os.path.join(opt.ori_dir, 'annotations')

    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir) if os.path.basename(path).find('.jpg')] 
    json_paths = [os.path.join(json_dir, path) for path in os.listdir(json_dir) if os.path.basename(path).find('json')]
    assert len(image_paths) != 0, 'there is no image in image_dir'
    assert len(json_paths) != 0, 'there is no json in json_dir'
    assert len(image_paths) == len(json_paths), 'inconsistent count of images and annotations'
    image_paths.sort()
    json_paths.sort()
    assert os.path.basename(image_paths[0]).replace('.jpg', '') == os.path.basename(json_paths[0]).replace('.json', ''), \
        'The name of image order is different from annotations'
    img_paths = split_tasks(image_paths, opt)
    lab_paths = split_tasks(json_paths, opt)
    json_paths = split_tasks(json_paths, opt)

    for i, task in enumerate(opt.tasks):
        output_lab_path = json2yolo(lab_paths[i], opt, task)
        output_img_path = copy_file(img_paths[i], opt, task, target_type='image')
        output_json_path = copy_file(json_paths[i], opt, task, target_type='json')

    return output_img_path, output_lab_path  # for visualization check


def check_output(img_path, lab_path, opt):
    name = os.path.basename(lab_path).replace('.txt', '')
    output_place = os.path.join(opt.check_dir, name+'_check.jpg')
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    with open(lab_path, 'r') as f:
        data = f.read().splitlines()
    for bbox in data:
        bbox = bbox.split(' ')[1:]  # omit class
        bbox = [float(cor) for cor in bbox]
        xmin = bbox[0] - bbox[2] / 2
        ymin = bbox[1] - bbox[3] / 2
        xmax = bbox[0] + bbox[2] / 2
        ymax = bbox[1] + bbox[3] / 2
        xmin = round(xmin * w)
        ymin = round(ymin * h)
        xmax = round(xmax * w)
        ymax = round(ymax * h)
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    cv2.imwrite(output_place, img)
    print(f'Check visualization in {output_place}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_dir', type=str, help='original data directory')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--val_ratio', type=float, default=0.3)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--tasks', default=['train', 'val', 'test'])
    parser.add_argument('--check_dir', type=str, help='directory for visualization output result')
    opt = parser.parse_args()
    check_parser(opt)
    create_dir(opt)

    img_path, lab_path = preprocess_dataset(opt)
    check_output(img_path, lab_path, opt)

