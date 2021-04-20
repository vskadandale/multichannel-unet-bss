#  Multi-channel Unet for Music Source Separation

**Note 1**: The pre-trained weights of most of the models used in these experiments are made available here: [https://shorturl.at/aryOX](https://shorturl.at/aryOX)

**Note 2**: For demos, visit our [project webpage](https://vskadandale.github.io/multi-channel-unet/).

#### <ins>Usage Instructions</ins>:
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
  Firstly, download the musdb dataset and obtain the samples in .wav format using the command [musdbconvert](https://pypi.org/project/musdb/). Once we have the .wav files in the folder corresponding to MUSDB_WAVS_FOLDER_PATH in settings.py, run the downsample_gt.py script to save a copy of downsampled wav files which will be used as a reference to compute the metrics during evaluation. Run the preprocessing.py script to generate train/val data splits and to convert the .wav samples to spectrograms. To get the track-wise energy profile, run the script compute_energy.py. Now, run filter_musdb_split.py with TYPE = '4src' as well as TYPE = '2src' setting so as to create lists of samples with non-silent sources for both the settings. 
  
  - Models are listed under : code/models.
  ```
  └── models
      ├── wrapper.py
      └── cunet.py
  ```
  The wrapper.py, as the name suggests serves as a wrapper for models used in the experiments. cunet.py contains the code for the Conditioned-U-Net model. Note that we do not have a separately written code for U-Net model because we U-Net is provided by the flerken-nightly 0.4.post10 package.  
  
  - Scripts for training/testing a new model are here: code/train or code/test
  ```
  └── train/test
      ├── baseline.py
      ├── cunet.py
      ├── dwa.py
      ├── unit_weighted.py
      └── energy_based.py
  ```
  Note that the training experiments need to be run after configuring the settings.py file accordingly. We provided code for baseline.py (dedicated u-nets), cunet.py (Conditioned U-Net), dwa.py (Dynamic Weight Average), and so on. Likewise to test a model, configure the weights path in settings.py along with other options listed there.
  
  - Scripts for evaluating a model are here: code/eval
    ```
    └── eval
        ├── stitch_audio.py
        └── eval_metrics.py
    ```
  Again, the settings.py file needs to be configured carefully before running these files. These scripts should be run after testing a model by running a script in code/test folder. stitch_audio.py stitches together the fragments of 6s audio estimated during model testing to form full-length track estimates. Then eval_metrics.py needs to be run to determine the performance of a source separation model in terms of metrics - SDR, SAR and SIR for each full length track. The results are dumped in the dumps folder configured in the settings.py in .csv format. 
  
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

#### <ins>Outline</ins>:
We train a single Multitask-U-Net for multi-instrument source separation using a weighted multi-task loss function. We investigate the source separation task in two settings: 1) singing voice separation (two sources), and 2) multi- instrument source separation (four sources). The number of final output channels of our U-Net corresponds to the total number of sources in the chosen setting. Each loss term in our multi-task loss function corresponds to the loss on the respective source estimates. We explore Dynamic Weighted Average (DWA) and Energy Based Weighting (EBW) strategies to determine the weights for our multi-task loss function. We compare the performance of our U-Net trained with the multi-task loss with that of dedicated U-Nets and the Conditioned-U-Net. Then we investigate the effect of training with the silent-source samples on the performance. We also study the effect of the choice of loss term definition on the source separation performance.

For more details, please check the pre-print [https://arxiv.org/abs/2003.10414](https://arxiv.org/abs/2003.10414).

#### <ins>Acknowledgements</ins>:
The data processing portion of the code borrows heavily from [https://github.com/hangzhaomit/Sound-of-Pixels](https://github.com/hangzhaomit/Sound-of-Pixels).

#### <ins>References</ins>:

[1] *G. Meseguer-Brocal, and G. Peeters. CONDITIONED-U-NET: Introducing a Control Mechanism in the U-net For Multiple Source Separations. In Proc. of ISMIR (International Society for Music Information Retrieval), Delft, Netherlands, 2019.*

[2] *A.Jansson, N.Montecchio, R.Bittner, A.Kumar, T.Weyde, E. J. Humphrey. Singing voice separation with deep u-net convolutional networks. In Proc. of ISMIR (International Society for Music Information Retrieval), Suzhou, China, 2017.*


## Citation
If you find our work useful, please cite our work as follows:
```
@inproceedings{kadandale2020multi,
  title={Multi-channel U-Net for Music Source Separation},
  author={Kadandale, Venkatesh S and Montesinos, Juan F and Haro, Gloria and G{\'o}mez, Emilia},
  booktitle={2020 IEEE 22nd International Workshop on Multimedia Signal Processing (MMSP)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}
```
