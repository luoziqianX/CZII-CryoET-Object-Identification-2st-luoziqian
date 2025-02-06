# 2st place CryoET challenge solution
This repo is the training part for CZII - CryoET Object Identification.
## Getting start
First clone and install dependencies
```bash
pip install -r requirements.txt
```
## Prepare data
1. You need to download the competition's data and set up the directory structure as shown below.
    ```
    .
    ├── compute-cv-7TTA.ipynb
    ├── copick_test.config
    ├── copick_utils
    │   ├── __about__.py
    │   ├── features
    │   ├── __init__.py
    │   ├── pickers
    │   ├── __pycache__
    │   ├── segmentation
    │   └── writers
    ├── data 
    │   ├── sample_submission.csv
    │   ├── test
    │   │   └── static
    │   └── train
    │       ├── overlay
    │       └── static
    ├── make-numpy-dataset.py
    ├── models
    │   ├── decoder.py
    │   ├── __init__.py
    │   ├── model2.py
    │   └── __pycache__
    ├── __pycache__
    │   ├── czii_helper.cpython-310.pyc
    │   ├── dataset.cpython-310.pyc
    │   ├── decoder.cpython-310.pyc
    │   └── model2.cpython-310.pyc
    ├── requirements.txt
    ├── submission.csv
    ├── train-SegResNet-6channel.py
    ├── train-Unet2E3D-6channel.py
    ├── train-Unet3D-6channel.py
    └── utils
        ├── czii_helper.py
        ├── dataset.py
        ├── __init__.py
        └── __pycache__
    ```
2. Make the numpy dataset for training
   ```bash
   python make-numpy-dataset.py 
   ```
## Training part
- To train SegResNet
  ```bash
  python train-SegResNet-6channel.py
  ```
- To train UNet6c
  ```bash
  python train-Unet3D-6channel.py
  ```
- To train UNet2E3D
  ```bash
  python train-Unet2E3D-6channel.py
  ```
## Compute CV Score for checkpoints
Due to the fluctuation of the loss in training.We prefer to compute every ckeckpoints' CV Score in 7TTA which is the same as submitions.You can run [compute-cv-7TTA.ipynb](https://github.com/luoziqianX/CZII-CryoET-Object-Identification-2st-luoziqian/blob/main/compute-cv-7TTA.ipynb) to compute the CV Score.You can replace part bellow to compute the ckpts you need and the numbers of the ckpts to run.
```python
path_dir = "./lightning_logs/version_412/checkpoints"
path_list = []
for fname in os.listdir(path_dir):
    if ".ckpt" in fname and "last" not in fname:
        path_list.append(os.path.join(path_dir, fname))
epoch_list = [path.split("=")[1].split("-")[0] for path in path_list]
epoch_list = [int(epoch) for epoch in epoch_list]
path_list.sort(key=lambda x: int(x.split("=")[1].split("-")[0]))
path_list = path_list[-20:]
```
We found that the one which get high CV Score both in training data and valid data will be more likely to get good score on LB.
## To ONNX & ONNX to TensorRT
We exported ONNX and convert ONNX to TensorRT both by kaggle notebook.Here are some examples.

- Exporting ONNX: [kaggle-notebooks/fork-of-fork-of-fork-of-7-model-6tta-t4x2-patch-no.ipynb](https://www.kaggle.com/code/luoziqian/fork-of-fork-of-fork-of-7-model-6tta-t4x2-patch-no)
- Converting ONNX to TensorRT: [kaggle-notebooks/fork-of-fork-of-fork-of-fork-of-model2trt-d93817.ipynb](https://www.kaggle.com/code/luoziqian/fork-of-fork-of-fork-of-fork-of-model2trt-d93817)
## Inference kernel
- Pytorch half version: [kaggle-notebooks/fork-of-7-model-6tta-t4x2-patch-norm-v1-17a4b7.ipynb](https://www.kaggle.com/code/luoziqian/fork-of-7-model-6tta-t4x2-patch-norm-v1-17a4b7)
- TensorRT FP16 Version: [kaggle-notebooks/trt-sub-model-6tta-t4x2-patch-norm-v4.ipynb](https://www.kaggle.com/code/luoziqian/trt-sub-model-6tta-t4x2-patch-norm-v4)