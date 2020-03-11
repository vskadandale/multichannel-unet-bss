#  Multi-task Unet for Music Source Separation

More details coming soon!

The pre-trained weights of most of the models used in these experiments are made available here: [https://t.ly/P5DJw](https://t.ly/P5DJw)

Significant parts of the code borrows from SoP.

We are using musdb18 as our dataset. Use the following commands in the command line to extract the full length wave files from the dataset.
```
pip3 install musdb
musdbconvert $RAW_MUSDB_PATH $MUSDB_WAVS_FOLDER_PATH
```

#### References:

[1] *G. Meseguer-Brocal, and G. Peeters. CONDITIONED-U-NET: Introducing a Control Mechanism in the U-net For Multiple Source Separations. In Proc. of ISMIR (International Society for Music Information Retrieval), Delft, Netherlands, 2019.*

[2] *A.Jansson, N.Montecchio, R.Bittner, A.Kumar, T.Weyde, E. J. Humphrey. Singing voice separation with deep u-net convolutional networks. In Proc. of ISMIR (International Society for Music Information Retrieval), Suzhou, China, 2017.*
