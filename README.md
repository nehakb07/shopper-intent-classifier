# Online Shopper Purchase-Intent Classification

## a. Problem statement
Predict whether an online shopping session will end in a purchase (`Revenue` = 1) or not
(`Revenue` = 0), using session behaviour features. Five classification models are trained
on the same dataset, compared on six metrics, and demonstrated in a Streamlit web app.

## b. Dataset description
- **Source:** UCI Machine Learning Repository, Online Shoppers Purchasing Intention Dataset
  (https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Instances:** 12,330 sessions | **Input features:** 17 | **Target:** `Revenue` (binary)
- **Class balance:** 15.5% positive (purchase), 84.5% negative
- **Features:** page counts and durations (Administrative, Informational, ProductRelated),
  Google Analytics metrics (BounceRates, ExitRates, PageValues), SpecialDay, Month,
  OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend.
- **Preprocessing:** `Revenue` and `Weekend` converted to 0/1; `Month` and `VisitorType`
  one-hot encoded; numeric features standardized (all inside each saved sklearn Pipeline).
  80/20 stratified train/test split, random_state = 42.

## c. GitHub repository link
TODO: paste your repository URL here

## d. Models used

### Comparison table (test set)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8812 | 0.8877 | 0.7432 | 0.3560 | 0.4814 | 0.4603 |
| Decision Tree | 0.8962 | 0.9090 | 0.7114 | 0.5550 | 0.6235 | 0.5702 |
| KNN | 0.8739 | 0.7993 | 0.6715 | 0.3639 | 0.4720 | 0.4322 |
| Naive Bayes | 0.6736 | 0.7939 | 0.2941 | 0.7906 | 0.4287 | 0.3249 |
| Support Vector Machine | 0.8861 | 0.8725 | 0.6996 | 0.4634 | 0.5575 | 0.5090 |

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | TODO: write your own observation |
| Decision Tree | TODO: write your own observation |
| KNN | TODO: write your own observation |
| Naive Bayes | TODO: write your own observation |
| Support Vector Machine | TODO: write your own observation |
| **Overall winner for this dataset** | TODO: name the winner and justify it with your metrics |

## Streamlit app
TODO: paste your deployed Streamlit link here

## Repository structure
```
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
├── model/   (5 .joblib pipelines)
└── <this notebook>.ipynb
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
