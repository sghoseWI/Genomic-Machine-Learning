#Author: Saptarshi Ghose 

rm(list=ls())   

#Set working directory
setwd("/Users/saptarshighose/Downloads")

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("TCGAbiolinks", version = "3.8")

library(BiocManager)
library(TCGAbiolinks)

browseVignettes("TCGAbiolinks")

#Breast Cancer Data Query
bc_query <- GDCquery(project = c("TCGA-BRCA"),
                         data.category = "Transcriptome Profiling", 
                         data.type = 'Gene Expression Quantification',
                         experimental.strategy = "RNA-Seq",
                         workflow.type = "HTSeq - Counts",
                         legacy = FALSE)

#Ovarian Cancer Data Query  
ov_query <- GDCquery(project = c("TCGA-OV"),
                     data.category = "Transcriptome Profiling", 
                     data.type = 'Gene Expression Quantification',
                     experimental.strategy = "RNA-Seq",
                     workflow.type = "HTSeq - Counts",
                     legacy = FALSE)

#Download, Tranpose, and Write Breast Cancer Data to CSV
GDCdownload(bc_query, method = "api", directory = "maf")
maf <- GDCprepare(bc_query, directory = "maf")
geneExp <- SummarizedExperiment::assay(maf)
transposed_bc <- t(geneExp)
write.csv(transposed_bc, file = "bc_expression.csv")

#Download, Tranpose, and Write Ovarian Cancer Data to CSV
GDCdownload(ov_query, method = "api", directory = "maf")
maf1 <- GDCprepare(ov_query, directory = "maf")
geneExp1 <- SummarizedExperiment::assay(maf1)
transposed_ov <- t(geneExp1)
write.csv(transposed_ov, file = "ov_expression.csv")


