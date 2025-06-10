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
        
        cols_to_drop = ['Unnamed: 0', 'track_id', 'explicit', 'time_signature']
        
        df_clean.drop(columns=cols_to_drop, inplace=True)
        
        print (f"Dropped {cols_to_drop} as unecessary columns\nNo of columns dropped: {len(cols_to_drop)}")
        print (f"Dataframe shape after cleaning:{df_clean.shape}")
        
        return df_clean
    
    df_clean = cleaning(df)
    
    #######################
    # Cleaning 
    #######################