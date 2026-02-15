import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Function to load and flatten images
def load_and_flatten_images(input_dir):
    all_features = []

    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)

        # Read image (grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # Skip unreadable files
        
        # Normalize to [0,1]
        img = img / 255.0

        # Flatten image into 1D array
        img_flattened = img.flatten()
        all_features.append(img_flattened)

    # Return as a 2D NumPy array: (num_images, num_pixels)
    return np.array(all_features[0]) # Remove or add [0] at end of all_features to compare all or just 1 image

# Load real dataset features
real_features = load_and_flatten_images("C:/my/dataset/real/values/location")

synthetic_features = load_and_flatten_images("G:/my/dataset/predicted/values/location")

# Flatten datasets for KS test and KDE
real_kde = real_features.flatten()
synthetic_kde = synthetic_features.flatten()

# Compute KS test
ks_stat, p_value = ks_2samp(real_kde, synthetic_kde)
print(f"KS Statistic: {ks_stat}, P-Value: {p_value}")

# KDE plot
plt.figure(figsize=(8, 6))
sns.kdeplot(real_kde, label="Real Dataset", fill=True)
sns.kdeplot(synthetic_kde, label="Synthetic Dataset", fill=True)
plt.xlabel("Feature Value")
plt.ylabel("Density")
plt.title("Diffusion Epoch 300 With Timesteps 850 Feature Distribution Comparison")
plt.legend()
plt.show()
