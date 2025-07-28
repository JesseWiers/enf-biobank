#!/usr/bin/env python3

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Paths
data_dir = "/projects/prjs1252/data_jesse_final_v3/nifti_dataset"
out_dir = "/projects/prjs1252/data_jesse_final_v3/nifti_dataset"
outlier_dir = "/projects/prjs1252/data_jesse_final_v3/nifti_outlier_plots"

# Ensure outlier plot directory exists
os.makedirs(outlier_dir, exist_ok=True)

# Grayscale values for classes 0, 1, 2, 3
grayscale_values = [0.0, 0.33, 0.67, 1.0]

# Cutoffs for outlier exclusion
width_max = 76.91
height_max = 70.33
width_min = 37.78     # not enforced (too-small kept)
height_min = 41.64    # not enforced (too-small kept)

# Target crop size
target_height = 71
target_width = 77

def process_patient(patient_dir):

    patient_path = os.path.join(data_dir, patient_dir)

    # Skip non-directories
    if not os.path.isdir(patient_path):
        return

    patient_out_dir = os.path.join(out_dir, patient_dir)
    img_out_path = os.path.join(patient_out_dir, "cropped_sa.nii.gz")


    mask_out_path = os.path.join(patient_out_dir, "cropped_seg_sa.nii.gz")

    if os.path.exists(img_out_path) and os.path.exists(mask_out_path):
        logging.info(f"✅ Cropped files already exist for {patient_dir}. Skipping.")
        return

    # Construct file paths
    img_path = os.path.join(patient_path, "sa.nii.gz")
    seg_path = os.path.join(patient_path, "seg_sa.nii.gz")

    # Skip missing files
    if not os.path.exists(img_path) or not os.path.exists(seg_path):
        logging.info(f"❌ Missing data for {patient_dir}. Skipping.")
        return

    # Load data
    cmr = nib.load(img_path)
    slices_images = cmr.get_fdata()
    cmr_seg = nib.load(seg_path)
    slices_masks = cmr_seg.get_fdata()

    assert slices_images.shape == slices_masks.shape, f"Shape mismatch for {patient_dir}"

    # Map labels to grayscale
    for class_label, grayscale_value in enumerate(grayscale_values):
        slices_masks[slices_masks == class_label] = grayscale_value

    max_area = 0
    max_area_z, max_area_t = 0, 0
    max_width = 0
    max_width_z, max_width_t = 0, 0
    max_height = 0
    max_height_z, max_height_t = 0, 0

    # Find largest bounding box
    for z in range(slices_masks.shape[2]):
        for t in range(slices_masks.shape[3]):
            slice_mask = slices_masks[:, :, z, t]

            nonzero_coords = np.where(slice_mask > 0)

            if nonzero_coords[0].size > 0 and nonzero_coords[1].size > 0:
                min_row, max_row = np.min(nonzero_coords[0]), np.max(nonzero_coords[0])
                min_col, max_col = np.min(nonzero_coords[1]), np.max(nonzero_coords[1])

                width = max_col - min_col
                height = max_row - min_row
                area = width * height

                if area > max_area:
                    max_area = area
                    max_area_z = z
                    max_area_t = t

                if width > max_width:
                    max_width = width
                    max_width_z = z
                    max_width_t = t

                if height > max_height:
                    max_height = height
                    max_height_z = z
                    max_height_t = t

    # Too narrow
    if max_width < width_min:
        save_outlier_plot(
            patient_dir,
            slices_images,
            slices_masks,
            max_width_z,
            max_width_t,
            f"(TOO NARROW) max_width = {max_width}"
        )
        logging.info(f"⚠️ Patient {patient_dir} has small width: {max_width}. Keeping data.")

    # Too short
    if max_height < height_min:
        save_outlier_plot(
            patient_dir,
            slices_images,
            slices_masks,
            max_height_z,
            max_height_t,
            f"(TOO SHORT) max_height = {max_height}"
        )
        logging.info(f"⚠️ Patient {patient_dir} has small height: {max_height}. Keeping data.")

    # Check for too wide
    if max_width > width_max:
        save_outlier_plot(patient_dir, slices_images, slices_masks, max_width_z, max_width_t,
                          f"(TOO WIDE) max_width = {max_width}")
        logging.info(f"⚠️ Skipping {patient_dir} due to excessive width: {max_width}")
        return

    # Check for too tall
    if max_height > height_max:
        save_outlier_plot(patient_dir, slices_images, slices_masks, max_height_z, max_height_t,
                          f"(TOO LONG) max_height = {max_height}")
        logging.info(f"⚠️ Skipping {patient_dir} due to excessive height: {max_height}")
        return

    # → **too-small segmentations are kept** ←




    # Compute final bounding box
    slice_mask = slices_masks[:, :, max_area_z, max_area_t]
    nonzero_coords = np.where(slice_mask > 0)

    min_row, max_row = np.min(nonzero_coords[0]), np.max(nonzero_coords[0])
    min_col, max_col = np.min(nonzero_coords[1]), np.max(nonzero_coords[1])

    bbox_height = max_row - min_row + 1
    bbox_width = max_col - min_col + 1

    extra_height = target_height - bbox_height
    extra_width = target_width - bbox_width

    if extra_height < 0 or extra_width < 0:
        raise ValueError(f"Bounding box exceeds target size for patient {patient_dir}")

    expand_top = extra_height // 2
    expand_bottom = extra_height - expand_top
    expand_left = extra_width // 2
    expand_right = extra_width - expand_left

    new_min_row = max(min_row - expand_top, 0)
    new_max_row = min(max_row + expand_bottom, slices_images.shape[0] - 1)
    new_min_col = max(min_col - expand_left, 0)
    new_max_col = min(max_col + expand_right, slices_images.shape[1] - 1)

    # Crop volumes
    cropped_slices_images = slices_images[
        new_min_row:new_max_row + 1,
        new_min_col:new_max_col + 1,
        :,
        :
    ]

    cropped_slices_masks = slices_masks[
        new_min_row:new_max_row + 1,
        new_min_col:new_max_col + 1,
        :,
        :
    ]

    # Prepare save paths
    patient_out_dir = os.path.join(out_dir, patient_dir)
    os.makedirs(patient_out_dir, exist_ok=True)

    img_out_path = os.path.join(patient_out_dir, "cropped_sa.nii.gz")
    mask_out_path = os.path.join(patient_out_dir, "cropped_seg_sa.nii.gz")

    # Save cropped images
    cropped_slices_images_nii = nib.Nifti1Image(
        cropped_slices_images.astype(cmr.get_data_dtype()), cmr.affine, cmr.header
    )
    cropped_slices_masks_nii = nib.Nifti1Image(
        cropped_slices_masks.astype(cmr_seg.get_data_dtype()), cmr_seg.affine, cmr_seg.header
    )

    nib.save(cropped_slices_images_nii, img_out_path)
    nib.save(cropped_slices_masks_nii, mask_out_path)

    # print(f"✅ Saved cropped data for {patient_dir}")

def save_outlier_plot(patient_dir, slices_images, slices_masks, z, t, title):
    """Save overlay plot for outlier slices."""
    slice_img = slices_images[:, :, z, t]
    slice_mask = slices_masks[:, :, z, t]

    fig, ax = plt.subplots(1, 1)
    ax.imshow(slice_img, cmap='gray')
    ax.imshow(slice_mask, cmap='viridis', alpha=0.5)
    ax.axis('off')
    plt.title(f"{title}\nPatient {patient_dir}")
    
    plot_path = os.path.join(outlier_dir, f"{patient_dir}_{title.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.savefig(plot_path)
    plt.close()

def main():
    patient_dirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    for patient_dir in tqdm(patient_dirs, desc="Processing Patients"):
        start_time = time.time()
        try:
            process_patient(patient_dir)
        except Exception as e:
            logging.warning(f"❌ Failed to process {patient_dir}: {str(e)}")
        elapsed = time.time() - start_time
        logging.info(f"Finished patient {patient_dir} in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
