!pip install rasterio




from google.colab import drive
drive.mount('/content/drive')



import os
import numpy as np
import rasterio
from scipy.ndimage import median_filter
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Paths
tif_folder = "/content/drive/My Drive/tif"
output_folder = os.path.join(tif_folder, "filtered_median")
os.makedirs(output_folder, exist_ok=True)

# Parameters
filter_size = 3  # Size of the median filter window (3x3)
mssim_scores = []

# Get first 10 TIFF files
tif_files = sorted([f for f in os.listdir(tif_folder) if f.endswith(".tif")])[:10]

# For displaying
plt.figure(figsize=(12, len(tif_files) * 4))

for idx, filename in enumerate(tif_files):
    filepath = os.path.join(tif_folder, filename)
    
    with rasterio.open(filepath) as src:
        bands = src.read()  # Shape: (bands, height, width)
        profile = src.profile

    filtered_bands = np.zeros_like(bands)
    ssim_values = []

    # Apply Median filter band-wise and compute SSIM
    for i in range(bands.shape[0]):
        original = bands[i].astype(np.float32)
        filtered = median_filter(original, size=filter_size)
        filtered_bands[i] = filtered

        # Normalize for SSIM
        min_val = original.min()
        max_val = original.max()
        if max_val - min_val != 0:
            original_norm = (original - min_val) / (max_val - min_val)
            filtered_norm = (filtered - min_val) / (max_val - min_val)
        else:
            original_norm = filtered_norm = original

        ssim_val = ssim(original_norm, filtered_norm, data_range=1.0)
        ssim_values.append(ssim_val)

    mean_ssim = np.mean(ssim_values)
    mssim_scores.append((filename, mean_ssim))

    # Save filtered image
    output_path = os.path.join(output_folder, filename.replace(".tif", "_median.tif"))
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(filtered_bands)

    # === Visualization: RGB ===
    def normalize_image(band_array):
        band_array = band_array.astype(np.float32)
        min_val, max_val = np.percentile(band_array, (1, 99))
        return np.clip((band_array - min_val) / (max_val - min_val + 1e-5), 0, 1)

    # Take first 3 bands as RGB
    if bands.shape[0] >= 3:
        orig_rgb = np.stack([
            normalize_image(bands[0]),
            normalize_image(bands[1]),
            normalize_image(bands[2])
        ], axis=-1)

        filt_rgb = np.stack([
            normalize_image(filtered_bands[0]),
            normalize_image(filtered_bands[1]),
            normalize_image(filtered_bands[2])
        ], axis=-1)

        # Display original RGB
        plt.subplot(len(tif_files), 2, 2 * idx + 1)
        plt.imshow(orig_rgb)
        plt.title(f"Original RGB - {filename}")
        plt.axis('off')

        # Display filtered RGB
        plt.subplot(len(tif_files), 2, 2 * idx + 2)
        plt.imshow(filt_rgb)
        plt.title(f"Median Filtered (MSSIM={mean_ssim:.4f})")
        plt.axis('off')

# Display all
plt.tight_layout()
plt.show()

# === MSSIM Summary ===
print("\n=== MSSIM Summary ===")
for fname, score in mssim_scores:
    print(f"{fname}: {score:.4f}")

# Average MSSIM
avg_mssim = np.mean([s[1] for s in mssim_scores])
print(f"\nAverage MSSIM for all 10 images: {avg_mssim:.4f}")
