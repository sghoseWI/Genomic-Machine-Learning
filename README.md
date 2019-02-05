# genomic-data-science

### Classifying Normal Samples and Tumors Samples from RNAseq profiles and Classifying Types of Tumors from RNAseq profiles

Used 5-fold cross validation and display the confusion matrix, f1 scores and other evaluation metrics in addition to the accuracy achieved.

There are two input files for each problem, one containing all the genes (60,483 features) and one containing just the protein coding genes (19,560 features).   You will likely want to use some kind of normalization for these features.

The input files are nt.coding.csv, nt.all.csv and type.coding.csv, type.all.csv

They can be found in the shared MLiC-Datasets-Lab1 directory.

There are two other files in that directory which map Ensembl IDs to Gene Symbol and Gene Types.  You might find those useful when developing gene signatures.  I’ll be adding a file that decodes the type IDs into Cancer types.  There are 18 types of cancer represented, a subset of the GDC types.



### Normal and Tumor Match Pair Analysis

a.	Using SciKit Learn build a machine learning classifier that takes RNAseq profiles from matched normal tumor pairs and classifies the sample as Normal or Tumor. Compare the nt.coding.csv vs the nt.all.csv.

b.	Using model selection methods of your choice determine which classical ML method performs best on the NT classification problem.

c.	Using feature selection methods of your choice determine a < 100 gene signature that can be used to classify Normal vs Tumor.

d.	Using Keras, build a deep learning classifier that performs the same classification task, and determine the learning curve (relationship of number of training samples to prediction accuracy) for your network, recommend using at least 10 training set sizes to estimate the learning curve.

e.	Use the TPOT autoML system to search for a better solution to part a.


### Cancer Type Classifier for 18 Common Tumor Types.

a.	Using SciKit Learn build a machine learning classifier that classifies Cancer Type from the type.coding.csv and type.all.csv files.  Compare the coding vs all genes cases.

b.	Using model selection methods of your choice, determine which classical ML method performs best.

c.	Using feature selection methods of your choice, determine a < 100 gene signature that can be used to classify tumor type.

d.	Using Keras, build a deep learning classifier that performs the same classification task, and determine the learning curve (relationship of number of training samples to prediction accuracy) for your network.

e.	Use the AutoKeras system to search for a better solution to part a.

f.	Use the Modal system to explore active learning on this problem.
