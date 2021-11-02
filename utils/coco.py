import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval


CATEGORY_ID = 0  # fixed ID


def check_parser(opt):
    assert opt.ann_dir
    assert Path(opt.ann_dir).is_dir()


def compute_mask_area(segm, h, w):
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_util.frPyObjects(segm, h, w)
        rle = mask_util.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = mask_util.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    
    mask = mask_util.decode(rle)
    area = mask.sum().item()

    return area


def convert_jsons_to_coco_format(opt):
    ann_dir = Path(opt.ann_dir)
    parent_dir = ann_dir.parents[0]
    img_dir = parent_dir / 'image'

    assert img_dir.exists()
    assert img_dir.is_dir()

    coco_json = {
        'images': [],
        'annotations': [],
        'categories': [{
            'id': CATEGORY_ID,
            'name': 'defect',
            'supercategory': 'Defect'
        }]
    }

    for i, ann_json_path in enumerate(tqdm(ann_dir.iterdir(), desc='merging jsons...')):
        name = ann_json_path.stem
        suffix = ann_json_path.suffix

        if suffix != '.json':
            continue

        img_path = Path.cwd() / img_dir / f'{name}.jpg'  # absolute path
        assert img_path.exists(), str(img_path)

        with ann_json_path.open('r') as f:
            json_str = f.read()
        ann_json = json.loads(json_str)

        segmentation = []
        for defect in ann_json['shapes']:
            points = defect['points']  # [[x1, y1], [x2, y2], ...]
            points = sum(points, [])  # [x1, y1, x2, y2, ...]
            segmentation.append(points)

        img_dict = {
            'id': i,
            'width': ann_json['imageWidth'],
            'height': ann_json['imageHeight'],
            'file_name': str(img_path)
        }
        coco_json['images'].append(img_dict)

        if segmentation:
            ann = {
                'id': i,
                'image_id': i,
                'category_id': CATEGORY_ID,
                'segmentation': segmentation,
                'area': compute_mask_area(segmentation, img_dict['height'], img_dict['width']),
                # 'bbox': []  # we do not need bbox,
                'iscrowd': 0
            }
            coco_json['annotations'].append(ann)

    coco_json = json.dumps(coco_json)
    coco_file_path = parent_dir / 'ann_coco.json'
    with coco_file_path.open('w') as f:
        f.write(coco_json)
    
    print(f'save coco file to {str(coco_file_path)}')


def check_coco(opt):
    ann_dir = Path(opt.ann_dir)
    parent_dir = ann_dir.parents[0]
    coco_file_path = parent_dir / 'ann_coco.json'

    assert coco_file_path.exists()

    coco = COCO(str(coco_file_path))
    print(f'find {len(coco.anns)} annotations')
    print('done')

    if opt.show_img:
        while True:
            img_id = np.random.choice(coco.getImgIds(), 1)[0].item()  # coco.loadImgs() can only accept a python number, not a numpy number
            img_dict = coco.loadImgs(img_id)[0]
            annIds = coco.getAnnIds(imgIds=img_dict['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)

            flag = False
            for ann in anns:
                if ann['segmentation']:  # not empty list
                    flag = True
            
            if flag:
                break

        img_path = img_dict['file_name']
        save_path = Path(img_path).parents[1] / 'sample.jpg'

        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(Path(img_path).name)
        plt.axis('off')
        coco.showAnns(anns)
        
        plt.savefig(str(save_path))


def get_img_id(coco, file_name):
    assert '/nfs' in file_name, '`file_name` is not an absolute path'

    for _, img_dict in coco.imgs.items():
        if img_dict['file_name'] == file_name:
            return img_dict['id']

    raise ValueError(f'Can not find the img ID of the img file {file_name}')


def bimask2rle(bimask):
    assert isinstance(bimask, np.ndarray)
    assert bimask.ndim == 2, f'expected 2-dim (h, w), but got {bimask.ndim}'

    rle = mask_util.encode(np.array(bimask[:, :, np.newaxis], order='F'))[0]

    return rle


def pred2coco(coco_gt, paths, bimasks, scores, result_file_path):
    # paths: [img_file_0, img_file_1, ...]
    # bimasks: [mask_0, mask_1, ...]
    # scores: [score_0, score_1, ...]

    assert len(paths) == len(bimasks) == len(scores), \
        f'#paths = {len(paths)}, #bimasks = {len(bimasks)}, #scores = {len(scores)}'

    json_results = []
    for path, bimask, score in tqdm(zip(paths, bimasks, scores), desc='generating result json'):
        img_id = get_img_id(coco_gt, path)
        rle = bimask2rle(bimask)
        rle['counts'] = rle['counts'].decode()  # in order to make it is JSON serializable

        data = {
            'image_id': img_id,
            'category_id': CATEGORY_ID,
            'segmentation': rle,
            'score': score
        }
        json_results.append(data)

    json_results = json.dumps(json_results)
    with open(result_file_path, 'w') as f:
        f.write(json_results)

    print(f'saved prediction to {result_file_path}')

    coco_dt = coco_gt.loadRes(result_file_path)

    return coco_dt


def coco_eval(coco_gt, coco_dt):
    img_ids = coco_gt.getImgIds()
    iou_type = 'segm'
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.imgIds = img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def generate_fake_predictions(coco_gt):
    paths = []
    bimasks = []
    scores = []

    for img_id in coco_gt.getImgIds():
        img_dict = coco_gt.loadImgs(img_id)[0]
        annIds = coco_gt.getAnnIds(imgIds=img_dict['id'], iscrowd=None)
        anns = coco_gt.loadAnns(annIds)

        if len(anns) > 0:  # has defect gt
            ann = anns[0]  # since only one class, len(anns) == 1
            bimask = coco_gt.annToMask(ann)
        else:
            shape = img_dict['height'], img_dict['width']
            bimask = np.zeros(shape, dtype=np.uint8)

        paths.append(img_dict['file_name'])
        bimasks.append(bimask)
        scores.append(np.random.choice(range(0, 100), 1).item())

    return paths, bimasks, scores


if __name__ == '__main__':
    # run this to generate a coco file of ground truth
    # for example,
    # python Defect_segmentation/yolov5/utils/coco.py --ann_dir './defect_data/green_crop/annotations'
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_dir', type=str, help='annotation directory')
    parser.add_argument('--show_img', action='store_true')
    opt = parser.parse_args()

    check_parser(opt)
    convert_jsons_to_coco_format(opt)
    check_coco(opt)


# if __name__ == '__main__':
#     # run this to generate fake prediction to check the result is good
#     # for example,
#     # python Defect_segmentation/yolov5/utils/coco.py
#     coco_path = '/nfs/Workspace/defect_data/green_crop/ann_coco.json'
#     result_file_path = '/nfs/Workspace/defect_data/green_crop/prediction_coco.json'

#     coco_gt = COCO(coco_path)
#     paths, bimasks, scores = generate_fake_predictions(coco_gt)
#     coco_dt = pred2coco(coco_gt, paths, bimasks, scores, result_file_path)
#     coco_eval(coco_gt, coco_dt)