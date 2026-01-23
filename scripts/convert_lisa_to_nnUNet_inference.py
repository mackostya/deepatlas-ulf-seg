import torchio as tio

from dataloaders import data_utils as dutil
import os
import nibabel as nib
import json
from pathlib import Path


def write_dataset_json(dataset_root: Path):
    """
    Create dataset.json for nnU-Net (v2 & v3 compatible).
    """
    dataset_root = Path(dataset_root)
    imagesTr = sorted((dataset_root / "imagesTr").glob("*0000.nii.gz"))
    imagesTs = sorted((dataset_root / "imagesTs").glob("*.nii.gz"))  # may be empty
    assert imagesTr, "No training images found!"

    # Build relative paths for JSON
    def rel(p):
        return str(p.relative_to(dataset_root))

    training = [{"image": rel(img), "label": rel(dataset_root / "labelsTr" / img.name)} for img in imagesTr]
    test = [rel(img) for img in imagesTs]

    json_dict = {
        "name": "LISA_LFSEG",
        "description": "T2-weighted brain MRI - 9 labels (hipp, extra ventricle, basal ganglia) for segmentation",
        "tensorImageSize": "4D",
        "reference": "",
        "licence": "",
        "release": "0.1",
        "labels": {
            "background": 0,
            "hipp_left": 1,
            "hipp_right": 2,
            "extra_vent_left": 3,
            "extra_vent_right": 4,
            "bg_caud_left": 5,
            "bg_caud_right": 6,
            "bg_lent_left": 7,
            "bg_lent_right": 8,
        },
        "channel_names": {
            "0": "T2",
            "1": "atlas_pauli",
            "2": "atlas_ho",
        },
        "file_ending": ".nii.gz",
        "numTraining": len(training),
    }

    with open(dataset_root / "dataset.json", "w") as f:
        json.dump(json_dict, f, indent=4)
    print(f"Wrote dataset.json with {len(training)} training and {len(test)} test cases.")


if __name__ == "__main__":
    data_path = dutil.initialize_data_path()

    images_files, atlas_files_paths = dutil.load_data_paths_inference(use_atlas=True)
    print(f"Processing inference data with {len(images_files)} images.")
    subjects = [
        tio.Subject(
            image=tio.ScalarImage(path=str(images_files[idx])),
            atlas=tio.ScalarImage(path=str(atlas_files_paths[idx])),
        )
        for idx in range(len(images_files))
    ]
    for subject in subjects:
        subject["image"].load()
        subject["atlas"].load()
    standard_transforms = dutil.get_standard_transforms_for_nnUNet()
    transforms = tio.Compose(standard_transforms)
    for i, subject in enumerate(subjects):
        print(f"Processing subject {subject['image'].path}")
        img_id = images_files[i].split("/")[-1].split("_")[2]
        subject_tr = transforms(subject)
        print(f"Transformed subject {i + 1}/{len(subjects)}: {img_id}")
        img_tensor = subject_tr["image"].data

        atlas_tensor = subject_tr["atlas"].data
        atlas_pauli_tensor = atlas_tensor.squeeze()[0]
        atlas_ho_tensor = atlas_tensor.squeeze()[1]
        # Prepare output paths
        print(f"Saving subject {img_id} to nnUNet format...")
        output_dir_imgs = os.path.join("./nnUNet_inference_files/")
        os.makedirs(output_dir_imgs, exist_ok=True)
        base_name_img = "LISA_" + img_id + "_0000"
        base_name_pauli_atlas = "LISA_" + img_id + "_0001"
        base_name_ho_atlas = "LISA_" + img_id + "_0002"

        # Save image
        img_nib = nib.Nifti1Image(img_tensor.squeeze().cpu().numpy(), affine=subject["image"].affine)
        img_path = os.path.join(output_dir_imgs, f"{base_name_img}.nii.gz")
        nib.save(img_nib, img_path)

        # Save atlas
        atlas_nib_pauli = nib.Nifti1Image(atlas_pauli_tensor.squeeze().cpu().numpy(), affine=subject["atlas"].affine)
        atlas_path_pauli = os.path.join(output_dir_imgs, f"{base_name_pauli_atlas}.nii.gz")
        nib.save(atlas_nib_pauli, atlas_path_pauli)

        # Save atlas
        atlas_nib_ho = nib.Nifti1Image(atlas_ho_tensor.squeeze().cpu().numpy(), affine=subject["atlas"].affine)
        atlas_path_ho = os.path.join(output_dir_imgs, f"{base_name_ho_atlas}.nii.gz")
        nib.save(atlas_nib_ho, atlas_path_ho)
    print("All subjects processed and saved in nnUNet format.")
