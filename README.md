# Working spaces
* jarvis: /usr/AI/Defect_segmentation
* mcut: /nfs/Workspace/Defect_segmentation

## TODO List
* Change to DenseConnect Module for segmentation
* Add segmentation augmentation in dataloader
* Try Dice loss, or tuning mask loss gain
* Support rect training or validation (masks and proto_out size should cautious)

## Action items
1. Dataloader (V)
2. Model (V)
3. Inference (V)
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
    For ignoring autoanchor, add `--noautoanchor`.
    * for mcut:
    ```
    python train.py --enable_seg True
    ```
    * for jarvis:
    ```
    python train.py --enable_seg True
    ```

## 1. Dataloader
### Process
* Done
### Issue

### Note
* json files are also splitted, and json paths are registered in `.yaml` also.
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
* Done

#### Issue
* We might need better refactor for the issue of changing index -1 to -2 for `Detect()` in original code.
* Using better seg module structure?

#### Note
* I use `model/yolov5s_seg.yaml` to configure the seg model

#### Target
* Output seg prediction map

#### Logic
* Add the segmentation model

### 2.2 Loss
DONE

#### Note
* the mask related hyperparameters are set in `yolov5/data/hyps/hyp.scratch.yaml`


## 3. Inference

### Issue
* Since our defects are plenty small, for visulization only, I draw the bbox slightly bigger then ground truth
* Originally, `rect` is opened in val dataset, which causes `masks` size is different from `proto_out`.
  therefore, I turn off `rect` in val dataset.
* The number of subplots is controled by batch_size, I change the batch-size to 1 for visualization both val and train
* train image visualization code is in `utils/loggers/__init__.py`

### Progress
At `train.py` line 398
    

## 4. Metric
1. FPS
2. common use mask mAP
    - [x] `val.py` uses the function `process_batch()` (line 212) to compute bbox IoU and `correct` which is collected in the list `stats` later, and then uses the function `utils.ap_per_class()` (line 236) to compute AP.
    - [ ] need to modify `process_batch()` to compute mask IoU instead of bbox IoU. Specifically, replace `box_iou()` in `process_batch()` with a function that can compute mask IoU.