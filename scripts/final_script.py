import glob
import os
import torchio as tio
import nibabel as nib
import subprocess
import typer
import shutil
from typing_extensions import Annotated
from pathlib import Path

from src import atlas_utils


nnunet_data_path = "/nnUNet_input/"
nnunet_output_path = "/nnUNet_output/"

def process_inference_data(images_files, pauli_atlas_files, ho_atlas_files):
    print(f"Processing inference data with {len(images_files)} images.")
    output_dir_imgs = os.path.join(nnunet_data_path)
    subjects = [
        tio.Subject(
            image=tio.ScalarImage(path=str(images_files[idx])),
            atlas_pauli=tio.ScalarImage(path=str(pauli_atlas_files[idx])),
            atlas_ho=tio.ScalarImage(path=str(ho_atlas_files[idx])),
        )
        for idx in range(len(images_files))
    ]

    for subject in subjects:
        subject["image"].load()
        subject["atlas_pauli"].load()
        subject["atlas_ho"].load()

    standard_transforms = [
        tio.ToOrientation("RAS"),
    ]
    transforms = tio.Compose(standard_transforms)

    os.makedirs(output_dir_imgs, exist_ok=True)
    for i, subject in enumerate(subjects):
        print(f"Processing subject {subject['image'].path}")
        img_id = images_files[i].split("/")[-1].split("_")[2]
        subject_tr = transforms(subject)
        print(f"Transformed subject {i + 1}/{len(subjects)}: {img_id}")
        img_tensor = subject_tr["image"].data

        atlas_pauli_tensor = subject_tr["atlas_pauli"].data
        atlas_ho_tensor = subject_tr["atlas_ho"].data
        # Prepare output paths
        print(f"Saving subject {img_id} to nnUNet format...")
        
        base_name_img = "LISA_" + img_id + "_0000"
        base_name_pauli_atlas = "LISA_" + img_id + "_0001"
        base_name_ho_atlas = "LISA_" + img_id + "_0002"

        # Save image
        img_nib = nib.Nifti1Image(img_tensor.squeeze().cpu().numpy(), affine=subject["image"].affine)
        img_path = os.path.join(output_dir_imgs, f"{base_name_img}.nii.gz")
        nib.save(img_nib, img_path)

        # Save atlas
        atlas_nib_pauli = nib.Nifti1Image(atlas_pauli_tensor.squeeze().cpu().numpy(), affine=subject["atlas_pauli"].affine)
        atlas_path_pauli = os.path.join(output_dir_imgs, f"{base_name_pauli_atlas}.nii.gz")
        nib.save(atlas_nib_pauli, atlas_path_pauli)

        # Save atlas
        atlas_nib_ho = nib.Nifti1Image(atlas_ho_tensor.squeeze().cpu().numpy(), affine=subject["atlas_ho"].affine)
        atlas_path_ho = os.path.join(output_dir_imgs, f"{base_name_ho_atlas}.nii.gz")
        nib.save(atlas_nib_ho, atlas_path_ho)
    print("All subjects processed and saved in nnUNet format.")
    return output_dir_imgs

def load_data_files(data_path):
    files = glob.glob(str(data_path / "**.nii.gz"))
    files_sorted = sorted(files)
    return files_sorted

def rename_nnunet_outputs(output_path):
    output_files = glob.glob(f"{nnunet_output_path}*nii.gz")
    for file_path in output_files:
        base_name = os.path.basename(file_path)
        # Extract the numeric ID from the filename (e.g., LISA_0001.nii.gz -> 0001)
        img_id = base_name.split("_")[1].split(".")[0]
        new_name = f"LISA_TESTING_SEG_{img_id}_baga.nii.gz" # _hipp - 2a _baga -2b
        new_path = os.path.join(output_path, new_name)
        os.makedirs(output_path, exist_ok=True)
        shutil.move(file_path, new_path)
        print(f"Renamed {file_path} -> {new_path}")

def main(    
        input_dir: Annotated[str, typer.Option()] = "/input",
        output_dir: Annotated[str, typer.Option()] = "/output",
        ):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")
    images_files = load_data_files(input_path)
    pauli_atlas_files, ho_atlas_files = atlas_utils.create_atlas_from_volumes(images_files)
    output_directory = process_inference_data(images_files, pauli_atlas_files, ho_atlas_files)
    print(f"Processed data saved to: {output_directory}")

    print("Setting up nnUNet environment variables...")
    os.environ["nnUNet_raw"] = "./nnUNet_atlas_full/nnUNet_raw"
    os.environ["nnUNet_preprocessed"] = "./nnUNet_atlas_full/nnUNet_preprocessed"
    os.environ["nnUNet_results"] = "./nnUNet_atlas_full/nnUNet_results"

    print("Running nnUNet inference...")
    subprocess.run(["nnUNetv2_predict", "-i", nnunet_data_path, "-o", nnunet_output_path, "-d", "1", "-c", "3d_fullres", "-npp", "0", "-nps", "0", "--disable_tta"])

    print("Renaming output files into LISA format...")
    
    rename_nnunet_outputs(output_path)
    print(f"Renamed outputs saved to: {output_path}")
    print("Finished!")

if __name__ == "__main__":
    typer.run(main)