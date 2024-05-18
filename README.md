# MDLD

## Introduction

This is code of MDLD (“Mask-Guided Target Node Feature Learning and Dynamic Detailed Feature Enhancement for LncRNA-Disease Association Prediction”).

## Dataset

| File_name                  | Data_type       | Source                                                       |Introduction                                |
| -------------------------- | --------------- | ------------------------------------------------------------ | -------------------------------------------|
| dis_sim_matrix_process.txt | disease-disease | [MeSH](https://www.nlm.nih.gov/mesh/meshhome.html)           |Similarity matrix of diseases               |
| lnc_dis_association.txt    | lncRNA-disease  | [LncRNADisease](https://www.cuilab.cn/lncrnadisease)         |Association matrix of diseases and lncRNA, where a value of 1 indicates a validated association, and 0 indicates no validated association|
| mi_dis.txt                 | miRNA-disease   | [HMDD](https://www.cuilab.cn/hmdd)                           |Association matrix of miRNA and diseases, where a value of 1 indicates a validated association, and 0 indicates no validated association|
| lnc_mi.txt                 | lncRNA-miRNA    | [starBase](https://rnasysu.com/encori/)                      |Interaction matrix of lncRNA and miRNA, where a value of 1 indicates a validated interaction, and 0 indicates no validated interaction|
| lnc_sim.txt                | lncRNA-lncRNA   | [Chen *et al.*](https://www.nature.com/articles/srep11338)$^{1}$ |Similarity matrix of lncRNA, obtained by calculating the similarity of diseases associated with two lncRNAs, dynamically calculated during the training process|
| lncRNA_name.txt                | lncRNA   | [LncRNADisease](https://www.cuilab.cn/lncrnadisease) |Name of the lncRNA|
| disease_name.txt                | disease   | [LncRNADisease](https://www.cuilab.cn/lncrnadisease) |Name of the disease|

(1) Chen, X., Clarence Yan, C., Luo, C. *et al.* Constructing lncRNA Functional Similarity Network Based on lncRNA-Disease Associations and Disease Semantic Similarity. *Sci Rep* **5**, 11338 (2015).

# File

```markdown
-utils : data preprocessing,parameters et al.					
-data : Dataset used in the research						
-models :  scripts for implementation of the model					
-main : scripts for model training and testing						
```

## Environment

```markdown
packages:
python == 3.9.0
torch == 1.13.0
numpy == 1.23.5
scikit-learn == 1.2.2
scipy == 1.10.1
pandas == 2.0.1
matplotlib == 3.7.1
```

# Run

```python
python ./main.py
```
