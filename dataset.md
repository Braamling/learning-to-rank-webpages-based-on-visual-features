### The ViTOR Dataset
*Disclaimer: This dataset may contain copyrighted material the use of which has not always been specifically authorized by the copyright owner. We are making such material available to enable research efforts in Information Retrieval. We believe this constitutes a 'fair use' of any such copyrighted material. The material in this dataset is distributed without profit to those who have expressed a prior interest in receiving the included information for research and educational purposes. We have provided links to certain datasets for reference purposes only and on an  ‘as is’ basis. You are solely responsible for your use of the datasets and for the complying with applicable terms and conditions, including any use restrictions and attribution requirements. We shall not be liable for, and specifically disclaim any warranties, express or implied, in connection with, the use of the datasets, including any warranties of fitness for a particular purpose or non-infringement.*

##### [jump to downloads](https://github.com/Braamling/learning-to-rank-webpages-based-on-visual-features/blob/master/dataset.md#download-vitor)
The ViTOR dataset consists of the contextual features, snapshots and highlighted snapshots used in `Learning to Rank Webpages Based on Visual Features by B. van den Akker et al.`. The data is organized as described below. The various directories can be downloaded separately. 
```
ViTOR
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
└───saliency
    │   clueweb12-0000tw-00-02137.png
    │   clueweb12-0000tw-01-09567.png
    │   ...
└───highlights
    │   201-clueweb12-0000tw-05-12114.png
    │   201-clueweb12-0000wb-30-01951.png
    |   ...

```

Each directory is filled as follows.

#### Features 
The `features` directory contains LETOR style formatted files with all document judgements and feature scores. The folder has 5 fold-partitions, which are used to create 5 folds that were used in the paper. The non-normalized features are included for reference. 
 
#### Snapshots
The `snapshots` directory contains the vanilla snapshots (a total of 28,488). Each images is indentified by its clueweb12 doc id (<CLUEBWEBID>.png)

#### Highlights 
The `highlights` directory contains the snapshots with their query depended red highlights (a total of 28,655). Each image is indentify with its TREC query id and clueweb12 doc id (<QUERYID>-<CLUEWEBID>.png).


#### Saliency 
The `saliency` directory contains the automatically generated saliency heatmaps (a total of 28,488). Each images is indentified by its clueweb12 doc id (<CLUEBWEBID>.png).


### Download ViTOR
Each of the directories can be downloaded separately: 
- [Features](https://drive.google.com/open?id=1Y9u-ADvM1mZH0-W8hWL4qdNfC3ONt-mE) (9mb). 
- [Snapshot images](https://drive.google.com/open?id=1KiurYA8_8tLNvx6xgjiteDpnUVbocOHf) (11gb).
- [Highlights images](https://drive.google.com/open?id=1BOdIu0FIC7X2PMfvvRzWJgZZ2wMXTt_S) (10gb).
- [Saliency images](https://drive.google.com/open?id=1xqNkuzrDQsUsAokM5Yi5t_--UCkz_1un) (451mb).

### Sources
- The contextual features and a subset of the screenshots are based on the documents in [ClueWeb12](https://lemurproject.org/clueweb12/).
- The relevance labels are taken from [TREC Web 2013](https://trec.nist.gov/data/web2013.html) & [TREC Web 2014](https://trec.nist.gov/data/web2014.html).
- A large subset of the screenshots have been taken from the [Internet Achive's Wayback Machine](https://archive.org/web/).

