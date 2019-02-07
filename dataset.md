### The ViTOR Dataset
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
