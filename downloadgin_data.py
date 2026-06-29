import shutil
import os

# Source path in the global cache folder
src_path = os.path.expanduser("~/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2")

# Destination path in your local project folder
dest_path = "kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2"
all_images_path = os.path.join(dest_path, 'all_images')

print("Creating local directories...")
os.makedirs(all_images_path, exist_ok=True)

# Copy the metadata CSV file to the local folder
csv_src = os.path.join(src_path, 'HAM10000_metadata.csv')
csv_dest = os.path.join(dest_path, 'HAM10000_metadata.csv')
if os.path.exists(csv_src):
    print("Copying metadata CSV...")
    shutil.copy(csv_src, csv_dest)

# Copy and combine images from both parts in the cache to the local all_images folder
for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
    folder_path = os.path.join(src_path, folder)
    if os.path.exists(folder_path):
        print(f"Copying images from {folder} to local all_images folder...")
        for file in os.listdir(folder_path):
            shutil.copy(os.path.join(folder_path, file), all_images_path)

print("Successfully copied all dataset files and images to your local project folder!")