# drainage-detection
## Problem Description
The main goal is to develop a model that, based on images obtained using spacecraft and unmanned aerial vehicles, will detect and segment the drainage system on agricultural land. The main goal is to create a model capable of identifying and classifying areas corresponding to drainage systems based on remote sensing data.

The programming stack:
- PyTorch
- Scikit-learn
- Albumentations
- Seaborn, Matplotlib


## Dataset Description
The dataset is represented by images of the fields in Menkovo (Saint-Petersburg). The images have shape: $256\times 256 \times 3$, where 3 means number of channels: RGB. The true images of drainage systems are represented by binary masks.

### Data Preprocessing
In order to split data into train and test data we needed to make stratification. The percentages of the drainage systems on image were counted for each image. Then each image was classified depending on the counted percentage. The class labels were used for stratification.

## Models and setting
The list of tested models: 
- DeepLabV3-mobilenet
- U-Net (tested with pretrained weights and without them)
- SegFormer-b1 (with pretrained weights on COCO dataset by Nvidia).

The setting were:
- learning rate was $10^{-4}$ with scheduler
- loss function: Cross-entropy
- batch size: 64
- number of epochs: 200
- optimizers: Adam, AdamW, SGD


## Metrics
The next metrics were counted in order to measure the quality of the models:
- Accuracy
- Precision
- Recall
- F1 score
- IoU
- Dice Score
- AUC-ROC

The most important metrics for Segmentation task are IoU and Dice Score.


## Results



| metric \ <br> model        | accuracy | precision | recall | f1    | IoU   | Dice-Score |
| --------------------- | -------- | --------- | ------ | ----- | ----- | ---------- |
| DeepLabV3             | 0.990    | 0.772     | 0.586  | 0.666 | 0.499 | 0.814      |
| UNet (pretrained)     | **0.994**    | 0.897     | 0.715  | 0.794 | 0.662 | 0.866      |
| UNet (not pretrained) | **0.994**    | **0.919**    | 0.700  | 0.793 | 0.659 | 0.872      |
| SegFormer             | **0.994**    | 0.853     | **0.751**  | **0.797** | **0.666** | **0.894**      |
