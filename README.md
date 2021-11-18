# Working spaces
* jarvis: /usr/AI/Defect_segmentation
* mcut: /nfs/Workspace/Defect_segmentation

## TODO List

### Model
1. Try Unet structure

### Dataloader
* Add segmentation augmentation in dataloader

### Loss
* Try Dice loss, or tuning mask loss gain

### Training
* Support rect training or validation (masks and proto_out size should cautious)
* Add attention module in Unet
* Change the upsample to transposeConv (x2 size, 1/2 channel)

## Action items
1. Dataloader (V)
2. Model (V)
3. Inference (V)
4. Metric (V)

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
    python train.py --enable_seg True --noautoanchor
    ```
    * for jarvis:
    ```
    python train.py --enable_seg True --noautoanchor
    ```

* Validation
    * In training:
        A ground truth annotation directory (validation data only) will be read from `data.yaml` (see the parser in `train.py`)
        A output `ann_coco.json` will be saved in the parent directory of the ground truth annotation directory 
        A prediction result json will be saved in `runs/exp_name/pred_coco.json`, which will be upadted per epoch
    
    * In validation only:
        A ground truth annotation file of COCO format is required. If there is no such file, please run this command to generate it.
        For example,

        ```
        python Defect_segmentation/yolov5/utils/coco.py --ann_dir './defect_data/green_crop/annotations' 
        ```

        The ground truth annotation file will be saved in the upper parent directory (i.e. `green_crop` in here).

        We can run the following command to evaluate and get mask mAP.

        * for mcut:
        ```
        python val.py --data [DATA] --weights [WEIGHTS] --ann-coco-path [ANN_COCO_PATH] --save-pred-coco [SAVE_PRED_COCO]
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
DONE
    

## 4. Metric
1. FPS
2. common use mask mAP
    - [x] `val.py` uses the function `process_batch()` (line 212) to compute bbox IoU and `correct` which is collected in the list `stats` later, and then uses the function `utils.ap_per_class()` (line 236) to compute AP.
