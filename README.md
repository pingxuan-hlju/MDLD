# MDLD

## Introduction

This is code of MDLD (“Mask-Guided Target Node Feature Learning and Dynamic Detailed Feature Enhancement for LncRNA-Disease Association Prediction”).

## Dataset

| File_name                  | Data_type       | Description                                                                                                                                                                                                                                         | Source                                                           |
|----------------------------|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| dis_sim_matrix_process.txt | disease-disease | The file contains the semantic similarities among 405 diseases. The value in _i_-th row and _j_-th column is the similarity between the _i_-th disease _d<sub>i</sub>_ and the _j_-th disease _d<sub>j</sub>_.                                      | [MeSH](https://www.nlm.nih.gov/mesh/meshhome.html)               |
| lnc_dis_association.txt    | lncRNA-disease  | The file includes the known 2687 associations between 240 lncRNAs and 405 diseases. The value in _i_-th row and _j_-th column is 1 when the _i_-th lncRNA _l<sub>i</sub>_ is associated with the _j_-th disease _d<sub>j</sub>_, otherwise it is 0. | [LncRNADisease](https://www.cuilab.cn/lncrnadisease)             |
| mi_dis.txt                 | miRNA-disease   | The file includes the known 13,559 associations between 495 miRNAs and 405 diseases. The value in _i_-th row and _j_-th column is 1 when the _i_-th miRNA _m<sub>i</sub>_ is associated with the _j_-th disease _d<sub>j</sub>_, otherwise it is 0. | [HMDD](https://www.cuilab.cn/hmdd)                               |
| lnc_mi.txt                 | lncRNA-miRNA    | The file includes the known 1002 interactions between 240 lncRNAs and 495 miRNAs. The value in _i_-th row and _j_-th column is 1 when the _i_-th lncRNA _l<sub>i</sub>_ is associated with the _j_-th miRNA _m<sub>j</sub>_, otherwise it is 0.     | [starBase](https://rnasysu.com/encori/)                          |
| lnc_sim.txt                | lncRNA-lncRNA   | The file contains the semantic similarities among 240 lncRNAs. The value in _i_-th row and _j_-th column is the similarity between the _i_-th lncRNA _l<sub>i</sub>_ and the _j_-th lncRNA _l<sub>j</sub>_.                                         | [Chen *et al.*](https://www.nature.com/articles/srep11338)$^{1}$ |
| lncRNA_name.txt            | lncRNA          | It contains the names of 240 lncRNAs.                                                                                                                                                                                                               | [LncRNADisease](https://www.cuilab.cn/lncrnadisease)             |
| disease_name.txt           | disease         | It contains the names of 405 diseases.                                                                                                                                                                                                              | [LncRNADisease](https://www.cuilab.cn/lncrnadisease)             |

(1) Chen, X., Clarence Yan, C., Luo, C. *et al.* Constructing lncRNA Functional Similarity Network Based on lncRNA-Disease Associations and Disease Semantic Similarity. *Sci Rep* **5**, 11338 (2015).

# File

```markdown
-utils : data preprocessing,parameters et al.					
-data : Dataset used in the research						
-models : scripts for implementation of the model					
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
