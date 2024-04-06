# MDLD

## Introduction

This is code of MDLD (“Mask-Guided Target Node Feature Learning and Dynamic Detailed Feature Enhancement for LncRNA-Disease Association Prediction”).

## Dataset

| File_name                  | Data_type       | Source                                                       |
| -------------------------- | --------------- | ------------------------------------------------------------ |
| dis_sim_matrix_process.txt | disease-disease | [MeSH](https://www.nlm.nih.gov/mesh/meshhome.html)           |
| lnc_dis_association.txt    | lncRNA-disease  | [LncRNADisease](https://www.cuilab.cn/lncrnadisease)         |
| mi_dis.txt                 | miRNA-disease   | [HMDD](https://www.cuilab.cn/hmdd)                           |
| lnc_mi.txt                 | lncRNA-miRNA    | [starBase](https://rnasysu.com/encori/)                      |
| lnc_sim.txt                | lncRNA-lncRNA   | [Chen *et al.*](https://www.nature.com/articles/srep11338)$^{1}$ |

(1) Chen, X., Clarence Yan, C., Luo, C. *et al.* Constructing lncRNA Functional Similarity Network Based on lncRNA-Disease Associations and Disease Semantic Similarity. *Sci Rep* **5**, 11338 (2015).

# File

```markdown
-utils : data preprocessing,parameters,experimental evaluation						
-data : data set						
-models : build model						
-main : model training and test						
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
