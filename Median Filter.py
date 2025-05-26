import os
import numpy as np
import rasterio
from scipy.ndimage import median_filter
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# ==== Paths ====
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
tif_folder = os.path.join(desktop, "images")
output_folder = os.path.join(desktop, "output_median")
os.makedirs(output_folder, exist_ok=True)

# ==== Parameters ====
filter_size = 3  # Median filter window size
mssim_scores = []

# ==== Load all TIFF files ====
tif_files = sorted([f for f in os.listdir(tif_folder) if f.lower().endswith(".tif")])

# ==== Display setup (only first 10 images) ====
plt.figure(figsize=(12, min(10, len(tif_files)) * 4))

for idx, filename in enumerate(tif_files):
    filepath = os.path.join(tif_folder, filename)

    with rasterio.open(filepath) as src:
        bands = src.read()  # Shape: (bands, height, width)
        profile = src.profile

    filtered_bands = np.zeros_like(bands)
    ssim_values = []

    # Apply median filter per band
    for i in range(bands.shape[0]):
        original = bands[i].astype(np.float32)
        filtered = median_filter(original, size=filter_size)
        filtered_bands[i] = filtered

        if idx < 10:
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

    if idx < 10:
        mean_ssim = np.mean(ssim_values)
        mssim_scores.append((filename, mean_ssim))

    # ==== Save filtered image with original name ====
    output_path = os.path.join(output_folder, filename)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(filtered_bands)

    # ==== Visualization (only for first 10 images) ====
    if idx < 10 and bands.shape[0] >= 3:
        def normalize_image(band_array):
            band_array = band_array.astype(np.float32)
            min_val, max_val = np.percentile(band_array, (1, 99))
            return np.clip((band_array - min_val) / (max_val - min_val + 1e-5), 0, 1)

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

        # Show original and filtered side by side
        plt.subplot(10, 2, 2 * idx + 1)
        plt.imshow(orig_rgb)
        plt.title(f"Original RGB - {filename}")
        plt.axis('off')

        plt.subplot(10, 2, 2 * idx + 2)
        plt.imshow(filt_rgb)
        plt.title(f"Median Filtered (MSSIM={mean_ssim:.4f})")
        plt.axis('off')

# ==== Show all plots ====
if len(tif_files) > 0:
    plt.tight_layout()
    plt.show()

# ==== MSSIM Summary ====
print("\n=== MSSIM Summary for First 10 Images ===")
for fname, score in mssim_scores:
    print(f"{fname}: {score:.4f}")

if mssim_scores:
    avg_mssim = np.mean([s[1] for s in mssim_scores])
    print(f"\nAverage MSSIM (First 10): {avg_mssim:.4f}")
else:
    print("No MSSIM computed (check input files).")
