#######################
# Import libraries
#######################
import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    print(f"Preprocessing script initialising...")
    
    # Load dataset into pandas DataFrame
    
    df = pd.read_csv("hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv")
    
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    
    #######################
    # Cleaning 
    #######################
    
    print(f"Dataframe cleaning...")
    
    def cleaning(df):
        df_clean = df.copy()
        
        df_clean.dropna()
        
        cols_to_drop = ['Unnamed: 0', 'artists', 'track_id', 'explicit', 'time_signature']
        
        df_clean.drop(columns=cols_to_drop, inplace=True)
        
        print (f"Dropped {cols_to_drop} as unecessary columns\nNo of columns dropped: {len(cols_to_drop)}")
        print (f"Dataframe shape after cleaning:{df_clean.shape}")
        
        return df_clean
    
    df_clean = cleaning(df)
    
    #######################
    # Target Feature Treatment 
    #######################
    
    print(f"Target feature treatment...")
    
    def target_treatment(df):
        
        df_target = df.copy()
        
        # Convert target feature to binary
        df_target['popularity'] = np.where(df_target['popularity'] >= 50, 1, 0)
        
        # Rename target feature
        df_target.rename(columns={'popularity': 'popularity_class'}, inplace=True)
        
        print(f"Target feature 'popularity' converted to binary and renamed to 'popularity_class")
        
        return df_target
    
    df_target = target_treatment(df_clean)
    
    
    #######################
    # Numeric Feature Treatment 
    #######################    