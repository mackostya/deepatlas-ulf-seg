import glob
import os
import numpy as np
import platform
import torchio as tio
import pandas as pd


def _get_sorted_file_paths(files, descriptor, separator):
    selected_files_flags = [descriptor in f for f in files]
    selected_files_paths = np.array(files)[selected_files_flags]
    files_descriptors = [f.split(separator)[-1] for f in selected_files_paths]
    idx_map = [files_descriptors.index(x) for x in sorted(files_descriptors)]
    sorted_files = np.array(selected_files_paths)[idx_map]
    return sorted_files


def load_all_file_paths(data_path):
    if platform.system() == "Windows":
        separator = "\\"  #
    elif platform.system() == "Linux":
        separator = "/"
    else:
        raise ValueError("Unsupported operating system. Only Windows and Linux are supported.")

    task2_subpath = "Task 2 - Segmentation/"
    all_files_task_2 = glob.glob(data_path + task2_subpath + "**/**.nii.gz")
    all_files_task_2_vent = glob.glob(data_path + task2_subpath + "Extra Segmentations/Ventricle/**.nii.gz")

    all_files = all_files_task_2 + all_files_task_2_vent
    # import target images
    images_files_paths_sorted = _get_sorted_file_paths(all_files, "ciso", separator)

    # import task 2a labels
    target_files_paths_sorted_hipp = _get_sorted_file_paths(all_files, "HF_hipp", separator)

    # import task 2b labels
    target_files_paths_sorted_baga = _get_sorted_file_paths(all_files, "HF_baga", separator)

    # import task extra labels
    target_files_paths_sorted_extra = _get_sorted_file_paths(all_files, "vent", separator)

    assert (
        len(target_files_paths_sorted_baga)
        == len(target_files_paths_sorted_hipp)
        == len(target_files_paths_sorted_baga)
        == len(target_files_paths_sorted_extra)
    )
    if len(images_files_paths_sorted) == 0:
        raise ValueError("No images found in the specified path")

    return (
        images_files_paths_sorted,
        target_files_paths_sorted_hipp,
        target_files_paths_sorted_baga,
        target_files_paths_sorted_extra,
    )


def initialize_data_path():
    if platform.system() == "Windows":
        data_path = "YOUR_DATA_PATH_HERE"
    elif platform.system() == "Linux":
        data_path = "YOUR_DATA_PATH_HERE"
    else:
        raise ValueError("Unsupported operating system. Only Windows and Linux are supported.")

    return data_path


def load_file_paths_from_split(data_path, split="train", use_atlas=False):

    if split not in ["train", "val"]:
        raise ValueError("Split must be either 'train' or 'val'.")
    df = pd.read_csv(os.getcwd() + f"/configs/{split}_data.csv")

    images_files_paths = df["images"].values
    target_files_paths_hipp = df["target_hipp"].values
    target_files_paths_baga = df["target_baga"].values
    target_files_paths_extra = df["target_extra"].values

    images_files_paths = [
        os.path.join(data_path + "Task 2 - Segmentation/Low Field Images", f) for f in images_files_paths
    ]
    target_files_paths_hipp = [
        os.path.join(data_path + "Task 2 - Segmentation/Subtask 2a - Hippocampus Segmentations", f)
        for f in target_files_paths_hipp
    ]
    target_files_paths_baga = [
        os.path.join(data_path + "Task 2 - Segmentation/Subtask 2b - Basal Ganglia Segmentations", f)
        for f in target_files_paths_baga
    ]
    target_files_paths_extra = [
        os.path.join(data_path + "Task 2 - Segmentation/Extra Segmentations/Ventricle", f)
        for f in target_files_paths_extra
    ]
    if use_atlas:
        atlas_files_paths = [os.path.join("YOUR_ATLAS_DATA_PATH", f) for f in df["atlas"].values]
        assert len(atlas_files_paths) == len(images_files_paths)

    assert (
        len(images_files_paths)
        == len(target_files_paths_hipp)
        == len(target_files_paths_baga)
        == len(target_files_paths_extra)
    )
    if len(images_files_paths) == 0:
        raise ValueError("No images found in the specified path")
    if use_atlas:
        return (
            images_files_paths,
            target_files_paths_hipp,
            target_files_paths_baga,
            target_files_paths_extra,
            atlas_files_paths,
        )
    return (
        images_files_paths,
        target_files_paths_hipp,
        target_files_paths_baga,
        target_files_paths_extra,
    )


def load_atlas_files(atlas_data_path="YOUR_ATLAS_DATA_PATH"):
    if platform.system() == "Windows":
        separator = "\\"  #
    elif platform.system() == "Linux":
        separator = "/"
    else:
        raise ValueError("Unsupported operating system. Only Windows and Linux are supported.")
    all_atlas_files = glob.glob(atlas_data_path + "**.nii.gz")
    all_atlas_files_sorted = _get_sorted_file_paths(all_atlas_files, separator=separator, descriptor="atlas")
    if len(all_atlas_files_sorted) == 0:
        raise ValueError("No atlas files found in the specified path")
    return all_atlas_files_sorted


def load_data_paths_inference(use_atlas=False):

    if platform.system() == "Windows":
        data_path = "YOUR_VALIDATION_DATA_PATH"
    elif platform.system() == "Linux":
        data_path = "YOUR_VALIDATION_DATA_PATH"
    else:
        raise ValueError("Unsupported operating system. Only Windows and Linux are supported.")

    all_files = glob.glob(data_path + "**.nii.gz")

    if len(all_files) == 0:
        raise ValueError("No files found in the specified path")
    if use_atlas:
        atlas_files = load_atlas_files(atlas_data_path="YOUR_ATLAS_DATA_PATH")
        all_files = _get_sorted_file_paths(all_files, "ciso", "/")
        return all_files, atlas_files
    return all_files


def get_standard_transforms(patch_size=(128, 128, 128), interpolate=True):

    standard_transforms = [
        tio.Resize(patch_size, image_interpolation="linear") if interpolate else tio.CropOrPad(patch_size),
        tio.ToOrientation("SAR"),
        tio.RescaleIntensity((0, 1)),  # re‐scale voxel values to [0,1]
        tio.ZNormalization(),  # per‐volume mean/std if you like
    ]
    return standard_transforms


def get_standard_transforms_for_nnUNet():

    standard_transforms = [
        tio.ToOrientation("RAS"),
    ]
    return standard_transforms
