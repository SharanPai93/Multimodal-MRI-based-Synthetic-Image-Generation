import cv2
import numpy as np
import os

def crop_black_border(img, threshold_value=10):
    """
    Crops out the black border from an image using a threshold.

    Args:
        img (numpy.ndarray): Input image (BGR).
        threshold_value (int): Pixel intensity cutoff for determining "non-black".

    Returns:
        numpy.ndarray: Cropped image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold: anything above 'threshold_value' becomes white, else black
    # Adjust 'threshold_value' as needed
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Find all contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # If no contours found, return the original image
        return img

    # We will compute the bounding box that contains all contours
    x_min, y_min = img.shape[1], img.shape[0]
    x_max, y_max = 0, 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x < x_min:
            x_min = x
        if y < y_min:
            y_min = y
        if x + w > x_max:
            x_max = x + w
        if y + h > y_max:
            y_max = y + h

    # Crop the image to that bounding box
    cropped_img = img[y_min:y_max, x_min:x_max]
    return cropped_img

def process_folder(input_folder, output_folder, threshold_value=10):
    """
    Processes all images in 'input_folder', cropping out black borders,
    and saves them to 'output_folder'.
    """
    os.makedirs(output_folder,exist_ok=True)

    # Loop over all image files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            in_path = os.path.join(input_folder, filename)
            out_path = os.path.join(output_folder, filename)

            # Read the image
            img = cv2.imread(in_path)
            if img is None:
                print(f"Warning: could not read {in_path}")
                continue

            # Crop black borders
            cropped_img = crop_black_border(img, threshold_value)

            # Save the result
            cv2.imwrite(out_path, cropped_img)
            print(f"Processed and saved: {out_path}")


if __name__ == "__main__":
    # Example usage:
    input_folder = "/content/drive/MyDrive/Kaggle_Images/healthy"
    output_folder = "/content/drive/MyDrive/Kaggle_Images/healthy_combined_cropped"

    # Adjust threshold_value if you find it's cropping too aggressively or not enough
    process_folder(input_folder, output_folder, threshold_value=15)
