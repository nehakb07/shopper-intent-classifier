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
https://github.com/nehakb07/shopper-intent-classifier

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
| Logistic Regression | It reached 0.881 accuracy and the highest precision of all five models (0.743), so most of the sessions it flags as purchases really are purchases. Its recall is only 0.356, so it misses most real buyers (246 of 382 in the test set). The train-test accuracy gap is just 0.004, so it generalizes well but is probably limited by its linear decision boundary. |
| Decision Tree | It performed best on accuracy (0.896), AUC (0.909), F1 (0.624) and MCC (0.570). Its recall of 0.555 is the second highest after Naive Bayes, so it finds many more buyers than Logistic Regression or KNN. With max_depth=6 the train-test gap is a modest 0.021, so some overfitting remains but is controlled. Trees can capture non-linear patterns and interactions between features. |
| KNN | It reached 0.874 accuracy, but its AUC (0.799) and MCC (0.432) are well below the top models, and its recall is only 0.364. It has the largest train-test gap (0.025). Distance-based methods tend to struggle with many features and with one-hot encoded columns, which probably hurts it here. |
| Naive Bayes | It has the lowest accuracy (0.674) and lowest MCC (0.325), but the highest recall (0.791), so it catches most buyers. Its precision is only 0.294, meaning it predicts "purchase" far too often and produces many false alarms. Some features are likely correlated, such as BounceRates and ExitRates, which would break its independence assumption. |
| Support Vector Machine | It was second best on F1 (0.558) and MCC (0.509), with accuracy of 0.886. Its recall (0.463) is better than Logistic Regression's, though its AUC (0.873) and precision are slightly lower. The RBF kernel lets it fit non-linear patterns that Logistic Regression can't, but it still trails the Decision Tree. |
| **Overall winner: Decision Tree** | It is best on four of the six metrics: accuracy, AUC, F1 and MCC. The classes are imbalanced (84.5% no purchase, 15.5% purchase), so MCC and F1 matter more than accuracy, and the Decision Tree leads both. If false alarms matter most, Logistic Regression has the best precision. If missing buyers is the bigger cost, Naive Bayes has the best recall. |

## Streamlit app
https://shopper-intent-ml-app.streamlit.app/

## Repository structure
```
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
├── model/   (5 .joblib pipelines)
└── ML_Assignment_2.ipynb
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
