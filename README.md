## Action items
1. Dataloader (V)
2. Model
3. Inference
4. Metric

### Code Snippet
* Preprocess data:
jarvis
```
python gt_preprocess.py --ori_dir /usr/AI/defect_data/green_crop --output_dir /usr/AI/defect_data/defect_seg_dataset --check_dir /usr/AI/defect_data/checker
```

mcut
```
python gt_preprocess.py --ori_dir /nfs/Workspace/defect_data/green_crop --output_dir /nfs/Workspace/defect_data/defect_seg_dataset --check_dir /nfs/Workspace/defect_data/checker
```


* Training
    * for mcut:
    ```
    python train.py --enable_seg True --json_dir /nfs/Workspace/defect_data/green_crop/annotations
    ```
    * for jarvis:
    ```
    python train.py --enable_seg True --json_dir /usr/AI/defect_data/green_crop/annotations/
    ```

## 1. Dataloader
### Process
* Done
### Issue

### Note
* The data
    * jarvis: `/usr/AI/defect_data/defect_seg_dataset`.
    * mcut: ``.
* I will turn off mosaic augmentation in dataloader save the effort
* One image with one mask, does not matter the instance counts.
* Remeber to use the same json files and data.yaml, they are decouple!
* In `yolov5/utils/gt_preprocess.py`, there is function for splitting dataset
* I have turned off all the data argumentation
* letterbox will not change mask, if img size is 640x640

### Target
Transform the mask data to (0/1) tensor

### Logic
First use sth like `json2yolov5` in `toyolov5_format.py` under `../Defect_detection/defect_data`, to get
1. image data
2. detection data
Let the mask data be read in the function in `dataset.py`

## 2. Model and Training

### 2.1 Model Construction

#### Process
In `yolo.py` and `common.py`.

#### Issue

#### Note

#### Target

#### Logic
Add the segmentation model after the module index

### 2.1 Training Script

## 3. Inference

## 4. Metric
1. FPS
2. common use mask mAP
    - [x] `val.py` uses the function `process_batch()` (line 212) to compute bbox IoU and `correct` which is collected in the list `stats` later, and then uses the function `utils.ap_per_class()` (line 236) to compute AP.
    - [ ] need to modify `process_batch()` to compute mask IoU instead of bbox IoU. Specifically, replace `box_iou()` in `process_batch()` with a function that can compute mask IoU.