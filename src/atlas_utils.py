import os
import tempfile
import nibabel as nib
import ants
import numpy as np
import glob
from brainextractor import BrainExtractor
from nilearn import datasets, image


def _write_labels_to_file(labels, file_path):
    with open(file_path, "w") as f:
        for line in labels:
            f.write(f"{line}\n")
    print(f"Labels written to {file_path}")


def _read_labels_from_file(file_path):
    with open(file_path, "r") as f:
        labels = [line.strip() for line in f]
    return labels


def load_atlas(extra_material_dir, template_path, atlas_type="pauli", download=False):
    if download:
        if atlas_type == "pauli":
            atlas = datasets.fetch_atlas_pauli_2017(version="prob")
        else:
            atlas = datasets.fetch_atlas_harvard_oxford("sub-prob-2mm")
        atlas_mask = atlas["maps"]  # 4D Nifti1Image: shape (x, y, z, N_regions)
        atlas_labels = atlas["labels"]
        template_nib = nib.load(template_path)
        atlas_mask = image.resample_to_img(atlas_mask, template_nib, interpolation="continuous")

        # Create background class
        atlas_data = atlas_mask.get_fdata()
        max_prob = np.max(atlas_data, axis=3)  # shape (X, Y, Z)
        background = 1.0 - max_prob
        background = np.clip(background, 0.0, 1.0)  # ensure non-negative
        # Both Pauli and Harvard Oxford atlases have 4D shape and do not have Background in atlas_mask, so we can concatenate the background
        atlas_with_bg = np.concatenate([background[..., np.newaxis], atlas_data], axis=3)
        atlas_with_bg_img = nib.Nifti1Image(atlas_with_bg, affine=atlas_mask.affine, header=atlas_mask.header)
        atlas_mask = image.math_img("np.argmax(imgs, axis=3)", imgs=atlas_with_bg_img)
        if atlas_type == "pauli":
            atlas_labels = ["Background"] + atlas_labels

        print(f"Atlas shape: {atlas_with_bg_img.shape}")
        if atlas_type == "pauli":
            nib.save(atlas_mask, os.path.join(extra_material_dir, "pauli_template_mask.nii.gz"))
            _write_labels_to_file(atlas_labels, os.path.join(extra_material_dir, "pauli_labels.txt"))
        else:
            nib.save(atlas_mask, os.path.join(extra_material_dir, "harvard_oxford_template_mask.nii.gz"))
            _write_labels_to_file(atlas_labels, os.path.join(extra_material_dir, "harvard_oxford_labels.txt"))
    else:
        if atlas_type == "pauli":
            atlas_mask = nib.load(os.path.join(extra_material_dir, "pauli_template_mask.nii.gz"))
            atlas_labels = _read_labels_from_file(os.path.join(extra_material_dir, "pauli_labels.txt"))
        else:
            atlas_mask = nib.load(os.path.join(extra_material_dir, "harvard_oxford_template_mask.nii.gz"))
            atlas_labels = _read_labels_from_file(os.path.join(extra_material_dir, "harvard_oxford_labels.txt"))

    return atlas_mask, atlas_labels


def save_nilearn_template_as_nifti(output_path):
    print("Fetching template using nilearn...")
    # template_nilearn_img = datasets.load_mni152_template()
    template_nilearn_img = nib.load(datasets.fetch_icbm152_2009()["t2"])
    nib.save(template_nilearn_img, output_path)
    print(f"Saved template to: {output_path}")


def load_template(template_dir, download=False):
    # 1. Get Template
    template_path = os.path.join(template_dir, "Template_152_T2.nii.gz")
    if (not os.path.exists(template_path)) and download:
        save_nilearn_template_as_nifti(template_path)
    template_ants = ants.image_read(template_path)
    return template_ants, template_path


def create_atlas_from_volumes(subject_img_paths):

    print(f"Creating atlas from {len(subject_img_paths)} subject images...")

    output_pauli_dir = "YOUR_PATH/atlas_files_pauli/"
    output_ho_dir = "YOUR_PATH/atlas_files_harvard_oxford/"
    os.makedirs(output_pauli_dir, exist_ok=True)
    os.makedirs(output_ho_dir, exist_ok=True)

    extra_material_dir = "./extra_material/"

    # 1. Load Template with Atlas
    if not os.path.exists(extra_material_dir):
        print("Creating extra material directory, downloading atlas...")
        download = True
        os.makedirs(extra_material_dir, exist_ok=True)
    else:
        print("Extra material directory already exists, not downloading atlas.")
        download = False

    template_ants, template_path = load_template(extra_material_dir, download=download)
    atlas_mask_p, atlas_labels_p = load_atlas(extra_material_dir, template_path, atlas_type="pauli", download=download)
    atlas_mask_ho, atlas_labels_ho = load_atlas(
        extra_material_dir, template_path, atlas_type="harvard_oxford", download=download
    )

    with tempfile.TemporaryDirectory(dir=extra_material_dir) as temp_dir:
        for subject_img_path in subject_img_paths:

            print(f"Processing subject image: {subject_img_path}")
            subject = ants.image_read(subject_img_path)
            subject_id = subject_img_path.split("/")[-1].split("_")[-2]

            # 2. Run the brain extraction on the subject image
            input_img = nib.load(subject_img_path)
            bet = BrainExtractor(img=input_img)

            # run the brain extraction
            print("Running brain extraction...")
            bet.run()
            brain_mask_path = os.path.join(temp_dir, "brain_mask.nii.gz")
            bet.save_mask(brain_mask_path)

            # 3. Mask the subject image
            print("Masking subject image...")
            subject_mask = ants.image_read(brain_mask_path)
            subject_brain = ants.mask_image(subject, subject_mask)

            print("Starting registration to the Template...")
            reg = ants.registration(
                fixed=template_ants,
                moving=subject_brain,
                type_of_transform="SyN",
                write_composite_transform=True,
            )
            bwdtransform = ants.read_transform(reg["invtransforms"])

            for atlas_type in ["pauli", "harvard_oxford"]:
                # 6. Apply backward transform to atlas mask
                output_dir = output_pauli_dir if atlas_type == "pauli" else output_ho_dir
                atlas_mask = atlas_mask_p if atlas_type == "pauli" else atlas_mask_ho
                atlas_labels = atlas_labels_p if atlas_type == "pauli" else atlas_labels_ho
                ants_mask = ants.image_read(atlas_mask.get_filename())

                atlas_mask_transformed = ants.apply_ants_transform(
                    bwdtransform,
                    ants_mask,
                    reference=subject_brain,
                    data_type="image",
                    interpolation="nearestNeighbor",
                )
                atlas_mask_transformed_bwd_path = os.path.join(temp_dir, f"atlas_mask.nii.gz")
                ants.image_write(atlas_mask_transformed, atlas_mask_transformed_bwd_path)
                atlas_mask_transformed = nib.load(atlas_mask_transformed_bwd_path)

                # 7. (Optional) Create atlas maps for specific labels
                if atlas_type == "pauli":
                    indexes_of_interest = [
                        atlas_labels.index("Background"),
                        atlas_labels.index("Pu"),
                        atlas_labels.index("Ca"),
                        atlas_labels.index("NAC"),
                    ]
                else:
                    indexes_of_interest = [
                        atlas_labels.index("Background"),
                        atlas_labels.index("Left Hippocampus"),
                        atlas_labels.index("Right Hippocampus"),
                        atlas_labels.index("Left Caudate"),
                        atlas_labels.index("Right Caudate"),
                        atlas_labels.index("Left Putamen"),
                        atlas_labels.index("Right Putamen"),
                    ]

                # Mask atlas_mask_transformed so only selected labels remain, others set to 0
                atlas_data = atlas_mask_transformed.get_fdata()
                mask = np.isin(atlas_data, indexes_of_interest)
                masked_data = np.where(mask, atlas_data, 0)

                # Save as a new Nifti image, preserving affine and header
                atlas_mask_transformed_stripped = nib.Nifti1Image(
                    masked_data, atlas_mask_transformed.affine, atlas_mask_transformed.header
                )

                if atlas_type == "pauli":
                    final_atlas_mask = atlas_mask_transformed
                else:
                    final_atlas_mask = atlas_mask_transformed_stripped
                final_atlas_mask_bwd_path = os.path.join(output_dir, f"atlas_mask_{subject_id}.nii.gz")
                final_atlas_mask_data_clipped = np.clip(final_atlas_mask.get_fdata(), 0, 1)
                template_nilearn_img = nib.Nifti1Image(
                    final_atlas_mask_data_clipped, final_atlas_mask.affine, final_atlas_mask.header
                )
                nib.save(template_nilearn_img, final_atlas_mask_bwd_path)
    pauli_files = glob.glob(output_pauli_dir + "*.nii.gz")
    ho_files = glob.glob(output_ho_dir + "*.nii.gz")
    pauli_files_sorted = sorted(pauli_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    ho_files_sorted = sorted(ho_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    print(f"Pauli Atlas files created: {len(pauli_files_sorted)}")
    print(f"Harvard Oxford Atlas files created: {len(ho_files_sorted)}")
    return pauli_files_sorted, ho_files_sorted
