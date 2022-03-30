# Identifying Transcription Factors prefer Methylated DNA using Reduced G-Gap Dipeptide Composition
#### Q. H. Nguyen, Hoang V. Tran, [B. P. Nguyen](https://people.wgtn.ac.nz/b.nguyen)∗, and Trang T. T. Do*


![Data representation for TF classification](TF.svg)
![Data representation for TFPM classification](TFPM.svg)

## Introduction
Transcription Factors (TFs) play an important role in gene expression and regulation of 3D genome conformation. TFs have ability to bind to specific DNA fragments called enhancers and promoters. Some TFs bind to promoter DNA fragments which are near the transcription initiation site and form complexes that allow polymerase enzymes to bind to the initiate transcription. Previous studies showed that methylated DNA has ability to inhibit and prevent TFs from binding to DNA fragments. However, recent studies have found that there are TFs that can bind to methylated DNA fragments. The identification of these TFs is an important steppingstone to a better understanding of cellular gene expression mechanisms. However, empirical methods are often time-consuming and labor-intensive, so developing machine learning methods is essential. In this study, we propose two machine learning methods for two problems: (1) classification of TFs, and (2) classification of TFs that prefer to bind to methylated DNAs (TFPMs) or non-methylated DNAs (TFPNM). For the TF classification problem, the proposed method uses the position-specific scoring matrix (PSSM) for data representation and a deep convolutional neural network (CNN) for modeling. This method achieved 90.56% sensitivity and 83.96% specificity on an independent test set. For the TFPM classification problem, the proposed method uses the reduced g-gap dipeptide composition for data representation and the support vector machine algorithm for modeling. This method achieved 82.61% sensitivity and 64.86% specificity on another independent test set. These results are higher than other studies on the same problems. 

## Availability and implementation
Please contact the corresponding authors for source code and data.

## Webserver
Please access our webserver [here](http://103.159.50.147)

## Contact 
[Go to contact information](https://homepages.ecs.vuw.ac.nz/~nguyenb5/contact.html)
