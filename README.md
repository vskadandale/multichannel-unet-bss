#  Multi-task Unet for Music Source Separation

**Note 1**: The pre-trained weights of most of the models used in these experiments are made available here: [https://t.ly/P5DJw](https://t.ly/P5DJw)

**Note 2**: The data processing portion of the code borrows heavily from [https://github.com/hangzhaomit/Sound-of-Pixels](https://github.com/hangzhaomit/Sound-of-Pixels).

#### Usage Instructions:
This repository is organized as follows:
The data preprocessing steps are covered under the folder: code/dataset. 
  ```
  └── dataset
      ├── compute_energy.py
      ├── dataloaders.py
      ├── downsample_gt.py
      ├── filter_musdb_split.py
      └── preprocessing.py
      
  ```
  
  - Models are listed under : code/models.
  ```
  └── models
      ├── wrapper.py
      └── cunet.py
  ```
  
  - Scripts for training/testing a new model are here: code/train or code/test
  ```
  └── train/test
      ├── baseline.py
      ├── cunet.py
      ├── dwa.py
      ├── unit_weighted.py
      └── energy_based.py
  ```
  
  - Scripts for evaluating a model are here: code/eval
    ```
    └── eval
        ├── stitch_audio.py
        └── eval_metrics.py
    ```
  
  - The various loss functions used in the experiments are here: code/loss
    ```
    └── loss
        └── losses.py
    ```
    
  - Other util files are put together under: code/utils
    ```
    └── utils
        ├── EarlyStopping.py
        ├── plots.py
        └── utils.py
    ```
    
  - settings.py is the file that hosts all the important configurations required to be set up before running the experiments.

#### Our Architecture:

#### References:

[1] *G. Meseguer-Brocal, and G. Peeters. CONDITIONED-U-NET: Introducing a Control Mechanism in the U-net For Multiple Source Separations. In Proc. of ISMIR (International Society for Music Information Retrieval), Delft, Netherlands, 2019.*

[2] *A.Jansson, N.Montecchio, R.Bittner, A.Kumar, T.Weyde, E. J. Humphrey. Singing voice separation with deep u-net convolutional networks. In Proc. of ISMIR (International Society for Music Information Retrieval), Suzhou, China, 2017.*

