import os
import nibabel as nib
import numpy as np

# Directories
pauli_dir = "/path/to/pauli_2017/"
harvard_dir = "/path/to/harvard_oxford/"
combined_dir = "/path/to/combined/"

# Ensure output directory exists
os.makedirs(combined_dir, exist_ok=True)

# Find all mask files in one directory (assuming both have the same files)
mask_files = sorted([f for f in os.listdir(pauli_dir) if f.startswith("atlas_mask_") and f.endswith(".nii.gz")])

for mask_file in mask_files:
    pauli_path = os.path.join(pauli_dir, mask_file)
    harvard_path = os.path.join(harvard_dir, mask_file)
    combined_path = os.path.join(combined_dir, mask_file)

    # Load masks
    pauli_img = nib.load(pauli_path)
    harvard_img = nib.load(harvard_path)
    pauli_data = pauli_img.get_fdata()
    harvard_data = harvard_img.get_fdata()

    # Check shapes
    if pauli_data.shape != harvard_data.shape:
        print(f"Shape mismatch for {mask_file}: {pauli_data.shape} vs {harvard_data.shape}")
        continue

    # Stack along a new last axis (channel)
    combined_data = np.stack([pauli_data, harvard_data], axis=-1)

    # Save combined mask
    combined_img = nib.Nifti1Image(combined_data, affine=pauli_img.affine, header=pauli_img.header)
    nib.save(combined_img, combined_path)
    print(f"Saved combined mask: {combined_path}")
