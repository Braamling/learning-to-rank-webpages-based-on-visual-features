### The ViTOR Dataset
*Disclaimer: This dataset contains copyrighted material the use of which has not always been specifically authorized by the copyright owner. We are making such material available to enable research efforts in Information Retrieval. We believe this constitutes a 'fair use' of any such copyrighted material as provided in Section 107 of the US Copyright Law. In accordance with Title 17 U.S.C. Section 107, the material in this dataset is distributed without profit to those who have expressed a prior interest in receiving the included information for research and educational purposes. If you wish to use copyrighted material in this dataset for purposes of your own that go beyond fair use, you must obtain permission from the copyright owner.*

##### [jump to downloads](https://github.com/Braamling/learning-to-rank-webpages-based-on-visual-features/blob/master/dataset.md#download-vitor)
The ViTOR dataset consists of the contextual features, snapshots and highlighted snapshots used in `Learning to Rank Webpages Based on Visual Features by B. van den Akker et al.`. The data is organized as described below. The Saliency images can either be used seperately 

```
ViTOR
│   README.txt  
│
└───features
│   │   normalized_set
│   │   Full_set
│   │   S1
│   │   S2
│   │   S3
│   │   S4
│   │   S5
│   │   readme.txt 
│   └───Fold1
│       │   test.txt
│       │   train.txt
│       │   vali.txt
│   └───Fold2
│       │   test.txt
│       │   train.txt
│       │   vali.txt
│   └───Fold3
│       │   test.txt
│       │   train.txt
│       │   vali.txt
│   └───Fold4
│       │   test.txt
│       │   train.txt
│       │   vali.txt
│   └───Fold5
│       │   test.txt
│       │   train.txt
│       │   vali.txt
│   
└───snapshots
    │   clueweb12-0000tw-00-02137.png
    │   clueweb12-0000tw-01-09567.png
    │   ...
└───highlights
    │   201-clueweb12-0000tw-05-12114.png
    │   201-clueweb12-0000wb-30-01951.png
    |   ...

```

### Download ViTOR
The full ViTOR dataset can be downloaded [here](https://drive.google.com/open?id=1afPX7fHmN6l4BUAJzj_CD3iSY7Ahk5Us) (21gb). The contextual features can also be downloaded seperately [here](https://drive.google.com/open?id=1Erp_GyY0-H9XQfDon4yasUG62xV5LKpG) (9mb).


### Download Additional resources

- The used saliency images can be downloaded [here](https://drive.google.com/open?id=1s286YhW0aC7qORUjpwR48tvUVUTz_OIM) (451mb)
- The actual HTML used to create the features can be requested [here](https://lemurproject.org/clueweb12/)
