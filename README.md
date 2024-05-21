# MDLD

## Introduction

This is code of MDLD (“Mask-Guided Target Node Feature Learning and Dynamic Detailed Feature Enhancement for LncRNA-Disease Association Prediction”).

## Dataset

| File_name                  | Data_type       | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Source                                                           |
|----------------------------|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| dis_sim_matrix_process.txt | disease-disease | This is a .txt file containing disease semantic similarity. It is a 405*405 matrix, where the values indicate the degree of similarity between two diseases.                                                                                                                                                                                                                                                                                                                                                    | [MeSH](https://www.nlm.nih.gov/mesh/meshhome.html)               |
| lnc_dis_association.txt    | lncRNA-disease  | This text file houses data on associations between lncRNAs and diseases. It presents this data in the form of a 240*405 matrix, which includes 2687 confirmed associations between lncRNAs and diseases. Each row in the data corresponds to the association details of a single lncRNA with 405 diseases, while each column corresponds to the association details of a single disease with 240 lncRNAs. An association is confirmed (represented by the value 1) or unconfirmed (represented by the value 0). | [LncRNADisease](https://www.cuilab.cn/lncrnadisease)             |
| mi_dis.txt                 | miRNA-disease   | This text file encapsulates data on associations between miRNAs and diseases. It is structured as a 495*405 matrix, which includes 13559 established miRNA-disease associations. Each row in the data corresponds to the association details of a single miRNA with 405 diseases, while each column corresponds to the association details of a single disease with 495 miRNAs. A value of 1 signifies a scientifically validated association, whereas 0 signifies an association that is yet to be confirmed.  | [HMDD](https://www.cuilab.cn/hmdd)                               |
| lnc_mi.txt                 | lncRNA-miRNA    | This is a .txt file that includes lncRNA-miRNA interaction data. It is a 240*495 matrix, containing 1002 confirmed lncRNA-miRNA interactions. Each row of the data represents whether there is a verified interaction between one lncRNA and 495 miRNAs, and each column represents the interaction information of one miRNA with 240 lncRNAs. A value of 1 indicates a confirmed interaction, while 0 indicates no experimentally verified interaction.                                                        | [starBase](https://rnasysu.com/encori/)                          |
| lnc_sim.txt                | lncRNA-lncRNA   | This text file contains the similarity data of lncRNAs. It is presented as a 240*240 matrix, where the values represent the similarity measure between two lncRNAs. It is calculated based on the diseases associated with the two lncRNAs during the data processing process.                                                                                                                                                                                                                                  | [Chen *et al.*](https://www.nature.com/articles/srep11338)$^{1}$ |
| lncRNA_name.txt            | lncRNA          | This is a .txt file containing the names of 240 lncRNAs.                                                                                                                                                                                                                                                                                                                                                                                                                                                        | [LncRNADisease](https://www.cuilab.cn/lncrnadisease)             |
| disease_name.txt           | disease         | This is a .txt file containing the names of 405 diseases.                                                                                                                                                                                                                                                                                                                                                                                                                                                       | [LncRNADisease](https://www.cuilab.cn/lncrnadisease)             |

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
