import kagglehub
import os
import shutil

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Download latest version
path = kagglehub.dataset_download("bhadramohit/climate-change-dataset")

print("Downloaded to:", path)

# Copy CSV files to the current directory
if os.path.exists(path):
    for file in os.listdir(path):
        if file.endswith('.csv'):
            source = os.path.join(path, file)
            destination = os.path.join(current_dir, file)
            shutil.copy2(source, destination)
            print(f"Copied {file} to current directory")

print("CSV files are now available in:", current_dir)