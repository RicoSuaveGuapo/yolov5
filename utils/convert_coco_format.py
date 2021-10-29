import json
import argparse
from tqdm import tqdm
from pathlib import Path

from pycocotools.coco import COCO


def check_parser(opt):
    assert opt.ann_dir
    assert Path(opt.ann_dir).is_dir()


def convert_jsons_to_coco_format(opt):
    ann_dir = Path(opt.ann_dir)
    parent_dir = ann_dir.parents[0]
    img_dir = parent_dir / 'image'

    assert img_dir.exists()
    assert img_dir.is_dir()

    coco_json = {
        'images': [],
        'annotations': []
    }

    for i, ann_json_path in enumerate(tqdm(ann_dir.iterdir(), desc='merging jsons...')):
        name = ann_json_path.stem
        suffix = ann_json_path.suffix

        if suffix != '.json':
            continue

        img_path = img_dir / f'{name}.jpg'
        assert img_path.exists(), str(img_path)

        with ann_json_path.open('r') as f:
            json_str = f.read()
        ann_json = json.loads(json_str)

        segmentation = []
        for defect in ann_json['shapes']:
            points = defect['points']  # [[x1, y1], [x2, y2], ...]
            points = sum(points, [])  # [x1, y1, x2, y2, ...]
            segmentation.append(points)

        coco_json['images'].append({
            'id': i,
            'width': ann_json['imageWidth'],
            'height': ann_json['imageHeight'],
            'file_name': str(img_path)
        })
        coco_json['annotations'].append({
            'id': i,
            'image_id': i,
            'category_id': 0,
            'segmentation': segmentation,
            # 'bbox': []  # we do not need bbox
        })

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


if __name__ == '__main__':
    # for example,
    # python Defect_segmentation/yolov5/utils/convert_coco_format.py --ann_dir './defect_data/green_crop/annotations'
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_dir', type=str, help='annotation directory')
    opt = parser.parse_args()

    check_parser(opt)
    convert_jsons_to_coco_format(opt)
    check_coco(opt)
