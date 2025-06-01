# BERT and LightGBM 

## Overview
This project is an implementation and leveraging the power of BERT and LightGBM together. It was tested on  binary classification dataset for Fake News. The approach involves integrating BERT embeddings into the system and enhancing its capabilities by incorporating gradient boosting through LightGBM. Additionally, the project includes a thorough examination of the model's performance in comparison to other traditional machine learning models using in combination with TF-IDF.

The projects inspiration came from reading this [research paper](https://link.springer.com/article/10.1007/s40747-023-01098-0).

## Dataset 
<a href='https://github.com/KaiDMML/FakeNewsNet/tree/maste'>
FakeNewsNet</a>

<a href="https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data">
ISOT </a>

## Main Idea
![alt text](image.png)

## Experiment

Result on ISOT Dataset
| **Model**           | **Accuracy %** | **Precision %** | **Recall %** | **F1 score %** |
|---------------------|----------------|-----------------|--------------|----------------|
| Decision Tree       | 91.11          | 91.13           | 91.11        | 91.11          |
| Naive Bayes         | 92.23          | 92.26           | 92.23        | 92.22          |
| Logistic Regression | 94.19          | 94.23           | 94.19        | 94.19          |
| LightGBM            | 93.34          | 93.53           | 93.43        | 94.43          |
| SVM                 | 94.20          | 94.25           | 94.20        | 94.20          |
| Random Forest       | 94.09          | 94.09           | 94.09        | 94.09          |
| BERT + LightGBM     | 98.63          | 98.79           | 98.55        | 98.67          |

