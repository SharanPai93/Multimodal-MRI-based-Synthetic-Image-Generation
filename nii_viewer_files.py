import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def convert_nii_to_images(source_folder, output_folder):
    """
    Converts all .nii or .nii.gz files in categorized subfolders into PNG images 
    and saves them in the respective output category folders.

    Parameters:
    - source_folder: The directory containing sorted .nii files (e.g., "Pediatric_Glioma_Sorted").
    - output_folder: The directory where extracted PNG images will be saved (e.g., "Pediatric_Glioma_Images").
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
    
    # Loop through each category folder (t1, t2_flair, t2, etc.)
    for category in os.listdir(source_folder):
        category_path = os.path.join(source_folder, category)
        if not os.path.isdir(category_path):
            continue  # Skip if not a directory

        output_category_folder = os.path.join(output_folder, category)
        os.makedirs(output_category_folder, exist_ok=True)  # Ensure category folder exists

        # Process each .nii/.nii.gz file in the category folder
        for file_name in os.listdir(category_path):
            if file_name.endswith(".nii") or file_name.endswith(".nii.gz"):
                nii_path = os.path.join(category_path, file_name)
                try:
                    # Load the MRI scan
                    img = nib.load(nii_path).get_fdata()
                    
                    # Normalize image (optional, but helps visualization)
                    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                    img = img.astype(np.uint8)

                    # Determine the number of slices and pick the middle slice
                    num_slices = img.shape[-1]  # Last axis is the depth (slices)
                    mid_slice = num_slices // 2  # Pick middle slice
                    
                    # Convert selected slices to images
                    for i, slice_idx in enumerate([mid_slice - 1, mid_slice, mid_slice + 1]):
                        if 0 <= slice_idx < num_slices:  # Ensure valid index
                            slice_img = img[:, :, slice_idx, :] if img.ndim == 4 else img[:, :, slice_idx]
                            
                            # Ensure RGB format: If it's single-channel, replicate across three channels
                            if slice_img.ndim == 2:  # Grayscale
                                slice_img = np.stack([slice_img] * 3, axis=-1)  # Convert to RGB (H, W, 3)

                            # Save the image
                            image_filename = f"{file_name.replace('.nii', '').replace('.gz', '')}_slice_{i}.png"
                            image_path = os.path.join(output_category_folder, image_filename)
                            plt.imsave(image_path, slice_img.astype(np.uint8))
                            print(f"Saved: {image_path}")
                except Exception as e:
                    print(f"Error processing {nii_path}: {e}")

        print(f"\n=====================================\n\
Done with category {category}\n\
=====================================\n\n")

# Example usage
source_folder = "C:/my/sorted/nii/files"  # Folder containing sorted nii files
output_folder = "C:/my/desination/folder"  # Folder where PNG images will be saved

convert_nii_to_images(source_folder, output_folder)
