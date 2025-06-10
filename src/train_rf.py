#######################
# Import libraries
#######################
import numpy as np
import pandas as pd
import time 
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

if __name__ == "__main__":
    
    np.random.seed(42)
    start = time.time()
    
    #######################
    # Fit/Train Model 
    #######################
    
    print(f"Random Forest Model training script initialising...")
    
    # Define datapath
    
    data_path = "data/spotify_tracks_numeric.csv" # Target path 
    
    # Load preprocesse.py dataset into pandas Dataframe
    
    dataframe = pd.read_csv(data_path)
    print(f"Dataset loaded with {dataframe.shape[0]} rows and {dataframe.shape[1]} columns")
    
    # Conduct 80/20 train/test split
    X = dataframe.drop(columns=['popularity_class'])
    y = dataframe['popularity_class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train/Test split completed: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    
    # Intialise Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Random Forest model initialised - training...")
    
    # Fit/Train model
    model.fit(X_train, y_train)
    print("Model training completed.")
    
    #######################
    # Model Preiction and Evaluation 
    #######################
    
    print("Model prediction and evaluation Output:\n")
    
    # Model Training Accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.2f}")
    
    # Model Testing Accuracy
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Testing Accuracy: {test_accuracy:.2f}")
    
    # Model F1 Score
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"F1 Score: {f1:.2f}")
    
    # Print Classification Report
    print("Classification Report:\n", classification_report(y_test, y_test_pred))

    #######################
    # Pickle Model
    #######################

    # Pickle the model for future usage (execution will create model directory if it does not exist)
    
    print("Pickling model...")
    pickle_path = "model/random_forest_model.pkl"
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    
    # Pickle model
    with open(pickle_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model pickled and saved to {pickle_path}.")
    
    end = time.time()
    print(f"Time taken for train_rf.py: {end - start:.2f} seconds")
    