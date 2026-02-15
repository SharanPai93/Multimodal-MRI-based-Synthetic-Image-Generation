import os
import shutil

def organize_nii_files(source_folder, destination_folder):
    """
    Loops through all subfolders in source_folder, identifies files containing 't1', 't2', etc. in their paths,
    and moves their .nii or .nii.gz files to corresponding folders in destination_folder.
    
    Parameters:
    - source_folder: Path to the main folder containing subfolders.
    - destination_folder: Path to the folder where sorted files will be placed.
    """
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    # Loop through all first-level subdirectories in the source folder
    for subfolder1 in os.listdir(source_folder):
        subfolder1_path = os.path.join(source_folder, subfolder1)

        if not os.path.isdir(subfolder1_path):  
            continue  # Skip if it's not a directory

        # Loop through all files (and possibly subfolders) inside subfolder1
        for file_name in os.listdir(subfolder1_path):
            file_path = os.path.join(subfolder1_path, file_name)  # Full path

            # Check if it's a .nii or .nii.gz file
            if file_name.endswith(".nii") or file_name.endswith(".nii.gz"):
                name = file_name.lower()
                # Categorize based on filename
                if "t1" in name:
                    category = "t1"
                elif "t2" in name and "flair" in name:
                    category = "t2_flair"
                elif "t2" in file_name.lower():
                    category = "t2"
                else:
                    category = "un_parsed"  # If no category matches

                # Destination path for this category
                category_folder = os.path.join(destination_folder, category)
                os.makedirs(category_folder, exist_ok=True)  # Create if it doesn't exist
                
                # Destination file path
                destination_file_path = os.path.join(category_folder, file_name)

                # Copy file to the appropriate category folder
                shutil.copy2(file_path, destination_file_path)
                print(f"Copied {file_name} to {category_folder}")

        print(f"=============================================\n\
Sorting for {subfolder1_path} complete!\n\
=============================================\n")

source_folder = "C:/my/source/folder"  # Change this to the actual source path
destination_folder = "C:/my/destination/folder"  # Change this to your desired destination path

organize_nii_files(source_folder, destination_folder)
