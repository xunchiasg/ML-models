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

A degree of emphasis was placed on the numerical features owing to their influence on ML model training, which required feature scaling and outlier treament. 

*Numerical Features with outliers*
![numerical_features_outliers](https://github.com/user-attachments/assets/9e13d61b-a76a-49ce-b31d-58f6ccc8bbc3)

*Features without outliers*
![numerical_features](https://github.com/user-attachments/assets/99250685-5b57-4238-b114-bc7328cef31e)

*Correlation Heatmap*
![corr_heatmap](https://github.com/user-attachments/assets/e55bf975-09c9-4660-b6f3-e90e000f25ec)


### Initial Model Training 

A total of 6 classification models were trained and baseline performance assessed based off test accuracy score and classification report metrices (precision, recall, F1-score) with a 3-fold cross validation applied. 

| Rank | Model                | Score              |
|------|----------------------|--------------------|
| 1    | **Random Forest**        | **0.8243950087375912** |
| 2    | **XGBoost**              | **0.7603920968804951** |
| 3    | **Decision Tree**        | **0.7493722182833807** |
| 4    | Gradient Boosting    | 0.7429138499621698 |
| 5    | Logistic Regression  | 0.7415541853292726 |
| 6    | K-Nearest Neighbors  | 0.730885222520016  |

Further 5-fold cross validation on the shortlisted top 3 models were conducted, with no significant accuracy drop confirming general model stability. 

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
- **Best Parameters Found**: 
  - `criterion`: 'gini', `max_depth`: 7, `max_features`: 'sqrt'
  - `min_samples_leaf`: 2, `min_samples_split`: 5, `n_estimators`: 200

However, the above proposed parameters ultimately led to generally degraded performance with a significant tendency towards Class 0 recall, indicating model overfitting towards the majority class.

| Model                        | Test Accuracy | Class 1 Recall | Class 1 F1 | Notes                              |
| ---------------------------- | ------------- | -------------- | ---------- | ---------------------------------- |
| **Random Forest (initial)**  | **0.8534**    | **0.49**       | **0.63**   | **Selected as Final Model**        |
| Random Forest (tuned)        | 0.7456        | 0.00           | 0.01       | Overfitting toward Class 0         |
| XGBoost (initial)            | 0.7807        | 0.26           | 0.37       | Lower recall than Random Forest    |
| XGBoost (tuned)              | 0.7678        | 0.16           | 0.26       | Performance degraded after tuning  |

Final model selection based on the problem statement was that of the **Initial Random Forest Model** for prediction of popularity class.

- **Key Rationale**:
  - Highest test accuracy (85.34%)
  - Best Class 1 recall (0.49) for identifying popular songs
  - Best Class 1 F1-score (0.63) showing balanced precision-recall
  - Stable performance without overfitting issues observed in tuned models

## Model Deployment Preparation
- **Next Step**: The final Random Forest model will be pickled using Python's `pickle` or library
- **Purpose**: To enable model deployment in production Python scripts
- **Usage**: The pickled model can be loaded and used for real-time song popularity predictions as-is

---
*(Updated June 6, 2025)*

