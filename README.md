# ML-models

This repository contains an exploratory data analysis of a spotify song dataset with the use of multiple Machine Learning algorithms and selection of the optimal algorithm to predict the popularity of a song. 

This entails the usage of classification models such as:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree 
- Random Forest
- Gradient Boosting
- XGBoost

## Dataset 

[spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)

The `spotify-tracks-dataset` is a dataset that is publicly available on Hugging Face that was extracted via the Spotify Web API. It contains songs over a range of 125 difference genres with associated audio features. 

Of note are the following features which are used in ML model training:

| Attribute         | Description |
|------------------|-------------|
| `duration_ms`     | The duration of the track in milliseconds. |
| `key`             | The musical key of a track, represented as an integer from 0 to 11, where 0 corresponds to C, 1 corresponds to C#/Db, 2 corresponds to D, and so on. |
| `mode`            | The mode of a track, which indicates whether it is in a major or minor key. A value of 0 indicates a minor key and a value of 1 indicates a major key. |
| `tempo`           | The tempo of a track in beats per minute (BPM). |
| `time_signature`  | The time signature of a track, represented as an integer indicating the number of beats per measure. |
| `valence`         | A measure of the positive or negative mood of a track, ranging from 0 (least positive) to 1 (most positive). |
| `danceability`    | A measure of how suitable a track is for dancing, based on its tempo, rhythm stability, and beat strength. |
| `energy`          | A measure of the intensity and activity of a track, ranging from 0 (least energetic) to 1 (most energetic). |
| `speechiness`     | A measure of the presence of spoken word in a track, ranging from 0 (least spoken word) to 1 (most spoken word). |
| `acousticness`    | A measure of the presence of acoustic instruments in a track, ranging from 0 (least acoustic) to 1 (most acoustic). |
| `liveness`        | A measure of the presence of a live audience in a track, ranging from 0 (least live) to 1 (most live). |
| `loudness`        | A measure of the perceived loudness of a track, in decibels (dB). |
| `instrumentalness`| A measure of the presence of vocals in a track, with a value ranging from 0 (least instrumental) to 1 (most instrumental). |

## Model Selection Process

### Data Transformation

Data exploration and associated steps towards final model evaluation and selection for an appropiate model for Class 1 (Popular Song) prediction were done in a Jupyter notebook environment. This involved data cleaning and transfomration processes towards the below target variable:

| popularity_class       | Encoded Value | Meaning   |
| ---------------------- | ------------- | --------- |
| 0–49                   | 0             | Unpopular |
| 50–100                 | 1             | Popular   |

A degree of emphasis was placed on the numerical features owing to their influence on ML model training, which required feature scaling and outlier treatment. 

*Nmerical features (outliers)*
![num_features_outliers](https://github.com/user-attachments/assets/c199ba92-8ac6-4770-8777-a9e6bc9e88f3)

- Image one 

- Image 2 

- Heatmap 

### Initial Model Training 

A total of 6 classification models were trained and baseline performance assessed based off test accuracy score and classification report metrices (precision, recall, F1-score) with a 3-fold cross validation applied. 

| Rank | Model                | Score              |
|------|----------------------|--------------------|
| 1    | **Random Forest**        | **0.8493** |
| 2    | **XGBoost**              | **0.7779** |
| 3    | **Decision Tree**        | **0.7590** |
| 4    | Gradient Boosting        | 0.760      |
| 5    | Logistic Regression      | 0.7451     |
| 6    | K-Nearest Neighbors      | 0.7425     |

Further 5-fold cross validation on the shortlisted top 3 models were conducted, with no significant accuracy drop confirming general model stability. 

| Model                | Accuracy Score (Mean over 3 Folds) |
|----------------------|------------------------------------|
| Random Forest        | 0.8219                             |
| XGBoost              | 0.7548                             |
| Decision Tree        | 0.7456                             |

### Hyperparamter Tuning

The top 2 models were subjected to hyperparamter tuning through GridsearchCV with a 5-fold cross validation in an attempt to get better model performance results.

 **Method**: GridSearchCV with 5-fold cross-validation
- **Models Tuned**: Random Forest and XGBoost
- **Random Forest Parameters**:
  - `n_estimators`: [100, 200]
  - `max_depth`: [3, 5, 7] 
  - `min_samples_split`: [2, 5]
  - `min_samples_leaf`: [1, 2]
  - `max_features`: ['sqrt', 'log2']
  - `criterion`: ['gini', 'entropy']

- **Best Parameters Proposed**: 
  - `criterion`: 'gini', `max_depth`: 7, `max_features`: 'sqrt'
  - `min_samples_leaf`: 2, `min_samples_split`: 2, `n_estimators`: 100

However, the above proposed parameters ultimately led to generally degraded performance with a significant tendency towards Class 0 recall, indicating model overfitting towards the majority class.

| Model                        | Test Accuracy | Class 1 Recall | Class 1 F1 | Notes                              |
| ---------------------------- | ------------- | -------------- | ---------- | ---------------------------------- |
| **Random Forest (initial)**  | **0.8509**    | **0.48**       | **0.62**   | **Selected as Final Model**        |
| Random Forest (tuned)        | 0.7458        | 0.01           | 0.01       | Severly degraded recall after tuning|
| XGBoost (initial)            | 0.7715        | 0.21           | 0.32       | Lower recall than Random Forest    |
| XGBoost (tuned)              | 0.7644        | 0.13           | 0.21       | Performance degraded after tuning  |

Final model selection based on the problem statement was that of the **Initial Random Forest Model** for prediction of popularity class.

- **Key Rationale**:
  - Highest test accuracy (85.09%)
  - Best Class 1 recall (0.48) for identifying popular songs
  - Best Class 1 F1-score (0.62) showing balanced precision-recall
  - Decent Class 1 performance without overfitting issues observed in tuned models

## ML Pipeline 
A pipeline to train the Random Forest model was created via `precprocess.py` and `train_rf.py` executed in the main python script. Execution of the script will also create a pickled model under a newly created `model` folder. 

**Model Classification Report**

```
Training Accuracy: 0.99
Testing Accuracy: 0.85
F1 Score: 0.83
Classification Report:
               precision    recall  f1-score   support

           0       0.85      0.98      0.91     17005
           1       0.89      0.48      0.62      5795

    accuracy                           0.85     22800
   macro avg       0.87      0.73      0.76     22800
weighted avg       0.86      0.85      0.83     22800
```

---
*(Updated June 11, 2025)*

