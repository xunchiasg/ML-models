#!/usr/bin/env python3
"""
Main script to run the ML pipeline for Spotify tracks classification.

This script orchestrates the execution of:
1. Data preprocessing (preprocess.py)
2. Model training (train_rf.py)
"""

import sys
import os
import subprocess
import time

def run_script(script_path, script_name):
    """
    Run a Python script and handle its execution.
    
    Args:
        script_path (str): Path to the script to run
        script_name (str): Name of the script for logging purposes
    
    Returns:
        bool: True if script ran successfully, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}")
    
    try:
        # Run the script using subprocess
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        
        print(f"\n✅ {script_name} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}:")
        print(f"Return code: {e.returncode}")
        return False
        
    except FileNotFoundError:
        print(f"\n❌ Script not found: {script_path}")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected error running {script_name}: {str(e)}")
        return False

def main():
    """
    Main function to orchestrate the ML pipeline execution.
    """
    print("🚀 Starting Spotify Tracks ML Pipeline")
    print(f"Current working directory: {os.getcwd()}")
    
    # Record start time
    pipeline_start = time.time()
    
    # Define scripts to run in order
    scripts = [
        ("src/preprocess.py", "Data Preprocessing"),
        ("src/train_rf.py", "Model Training")
    ]
    
    # Check if src directory exists
    if not os.path.exists("src"):
        print("❌ Error: 'src' directory not found in current working directory.")
        print("Please ensure you're running this script from the project root directory.")
        sys.exit(1)
    
    # Run each script in sequence
    success_count = 0
    total_scripts = len(scripts)
    
    for script_path, script_name in scripts:
        if not os.path.exists(script_path):
            print(f"❌ Warning: Script {script_path} not found. Skipping...")
            continue
            
        success = run_script(script_path, script_name)
        if success:
            success_count += 1
        else:
            print(f"\n⚠️  Pipeline halted due to error in {script_name}")
            break
    
    # Calculate total time
    pipeline_end = time.time()
    total_time = pipeline_end - pipeline_start
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Scripts completed successfully: {success_count}/{total_scripts}")
    print(f"Total pipeline time: {total_time:.2f} seconds")
    
    if success_count == total_scripts:
        print("🎉 All scripts completed successfully!")
        print("\nGenerated outputs:")
        if os.path.exists("data/spotify_tracks_numeric.csv"):
            print("  ✅ data/spotify_tracks_numeric.csv (preprocessed data)")
        if os.path.exists("model/random_forest_model.pkl"):
            print("  ✅ model/random_forest_model.pkl (trained model)")
    else:
        print("⚠️  Some scripts failed to complete. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
