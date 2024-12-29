# PGSD_WISS
## Dataset Preparation
* PASCAL VOC 2012
  ```python
  sh data/download_voc.sh
  ```
* COCO
  ```python
  sh data/download_voc.sh
  python data/coco/make_annotation.py
  ```
## Environment
* python (3.1)
* pytorch (1.6)
* [inplace-abn](https://github.com/mapillary/inplace_abn) (1.0.7)
* other environment requirements
  ```python
  pip install wandb

  conda install tensorboard
  conda install jupyter
  conda install matplotlib
  conda install tqdm
  conda install imageio
  ```
  
## Training
* Dowload pretrained model from [ResNet-101_iabn](https://github.com/arthurdouillard/CVPR2021_PLOP/releases/download/v1.0/resnet101_iabn_sync.pth.tar) to the ``pretrained`` folder.
  
* Train PASCAL VOC 2012
  ```python
  sh run.sh
  ```

* Train COCO
  ```python
  sh coco.sh
  ```
  
## Inference
* Modifying the bash file by adding ``--test`` will skip all the training procedures and test the model on test data.

## Contact
* If there are any questions, please feel free to contact with the author: Xu-Ze Hao (22210240018@m.fudan.edu.cn).
