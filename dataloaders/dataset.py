from torch.utils.data import Dataset

import torch
import torch.nn.functional as F
import torchio as tio
import dataloaders.data_utils as dutil


class LISADataset(Dataset):
    def __init__(
        self,
        patch_size=(128, 128, 128),
        split="train",
        transforms=None,
        use_atlas=False,
        interpolate=True,
    ):
        self.use_atlas = use_atlas
        self.patch_size = patch_size
        self.split = split
        self.interpolate = interpolate
        data_path = dutil.initialize_data_path()
        all_files = dutil.load_file_paths_from_split(data_path=data_path, split=split, use_atlas=use_atlas)
        if self.use_atlas:
            images_files, target_files_hipp, target_files_baga, target_files_extra, atlas_files = all_files
        else:
            images_files, target_files_hipp, target_files_baga, target_files_extra = all_files
            atlas_files = None
        self.images_files = images_files
        # Create Torchio Subjects
        self.subjects = [
            tio.Subject(
                image=tio.ScalarImage(path=str(self.images_files[idx])),
                label_h=tio.LabelMap(path=str(target_files_hipp[idx])),
                label_b=tio.LabelMap(path=str(target_files_baga[idx])),
                label_e=tio.LabelMap(path=str(target_files_extra[idx])),
                atlas=tio.LabelMap(path=str(atlas_files[idx])) if self.use_atlas else None,
            )
            for idx in range(len(self.images_files))
        ]

        # Load all images into RAM
        # First load moves the data to RAM, afterwards it is cached
        for subject in self.subjects:
            subject["image"].load()
            subject["label_h"].load()
            subject["label_b"].load()
            subject["label_e"].load()
            if use_atlas:
                subject["atlas"].load()
        self.set_tranforms(transforms)

    def set_tranforms(self, transforms=None):
        """Set transforms for the dataset."""
        standard_transforms = dutil.get_standard_transforms(patch_size=self.patch_size, interpolate=self.interpolate)

        if transforms:
            # Create Transform
            self.transform = tio.Compose(
                standard_transforms + transforms,
            )
        else:
            self.transform = tio.Compose(standard_transforms)

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, idx):

        subject = self.subjects[idx]
        original_spatial_shape = subject["image"].data.shape[1:]

        subject_tr = self.transform(subject)
        # from cache in RAM
        img_tensor = subject_tr["image"].data

        label_h_tensor = subject_tr["label_h"].data
        label_b_tensor = subject_tr["label_b"].data
        label_e_tensor = subject_tr["label_e"].data

        label_tensor = label_h_tensor + label_b_tensor + label_e_tensor
        label_tensor = label_tensor.clamp(0, 8).squeeze().long()
        if self.use_atlas:
            atlas_tensor = subject_tr["atlas"].data
            atlas_tensor = atlas_tensor.squeeze().long()
            return (img_tensor, label_tensor, torch.tensor(original_spatial_shape, dtype=torch.long), atlas_tensor)
        return (img_tensor, label_tensor, torch.tensor(original_spatial_shape, dtype=torch.long))


class LISAEvalDataset(Dataset):
    def __init__(
        self,
        patch_size=(128, 128, 128),
        split="val",
        use_atlas=False,
        interpolate=True,
    ):

        self.patch_size = patch_size
        self.split = split
        self.use_atlas = use_atlas
        self.interpolate = interpolate
        data_path = dutil.initialize_data_path()
        all_files = dutil.load_file_paths_from_split(data_path=data_path, split=split, use_atlas=use_atlas)
        if self.use_atlas:
            images_files, target_files_hipp, target_files_baga, target_files_extra, atlas_files = all_files
        else:
            images_files, target_files_hipp, target_files_baga, target_files_extra = all_files
            atlas_files = None
        self.images_files = images_files
        self.target_files_hipp = target_files_hipp
        self.target_files_baga = target_files_baga
        self.target_files_extra = target_files_extra
        # Create Torchio Subjects
        self.subjects = [
            tio.Subject(
                image=tio.ScalarImage(path=str(self.images_files[idx])),
                label_h=tio.LabelMap(path=str(target_files_hipp[idx])),
                label_b=tio.LabelMap(path=str(target_files_baga[idx])),
                label_e=tio.LabelMap(path=str(target_files_extra[idx])),
                atlas=tio.LabelMap(path=str(atlas_files[idx])) if self.use_atlas else None,
            )
            for idx in range(len(self.images_files))
        ]

        # Load all images into RAM
        # First load moves the data to RAM, afterwards it is cached
        for subject in self.subjects:
            subject["image"].load()
            subject["label_h"].load()
            subject["label_b"].load()
            subject["label_e"].load()
            if use_atlas:
                subject["atlas"].load()
        self.set_tranforms()

    def set_tranforms(self):
        """Set transforms for the dataset."""
        standard_transforms = dutil.get_standard_transforms(patch_size=self.patch_size, interpolate=self.interpolate)

        self.transform = tio.Compose(standard_transforms)

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, idx):

        subject = self.subjects[idx]
        original_spatial_shape = subject["image"].data.shape[1:]

        subject_tr = self.transform(subject)
        # from cache in RAM
        img_tensor = subject_tr["image"].data

        original_image_tensor = subject["image"].data

        label_h_tensor = subject["label_h"].data
        label_b_tensor = subject["label_b"].data
        label_e_tensor = subject["label_e"].data

        seg_hipp_one_hot = F.one_hot(label_h_tensor.long(), num_classes=9).movedim(-1, 1).squeeze()
        seg_bas_one_hot = F.one_hot(label_b_tensor.long(), num_classes=9).movedim(-1, 1).squeeze()
        seg_extra_one_hot = F.one_hot(label_e_tensor.long(), num_classes=9).movedim(-1, 1).squeeze()

        # compose all labels into one tensor
        # background is not use dfor the evaluation calculation
        original_label_tensor = torch.zeros_like(seg_hipp_one_hot)
        original_label_tensor[1:, ...] = (
            seg_hipp_one_hot[1:, ...] + seg_bas_one_hot[1:, ...] + seg_extra_one_hot[1:, ...]
        ).float()
        if self.use_atlas:
            atlas_tensor = subject_tr["atlas"].data
            atlas_tensor = atlas_tensor.squeeze().long()
            return (
                img_tensor,
                original_image_tensor,
                original_label_tensor,
                atlas_tensor,
                torch.tensor(original_spatial_shape, dtype=torch.long),
                [
                    self.images_files[idx],
                    self.target_files_hipp[idx],
                    self.target_files_baga[idx],
                    self.target_files_extra[idx],
                ],
            )
        else:
            return (
                img_tensor,
                original_image_tensor,
                original_label_tensor,
                torch.tensor(original_spatial_shape, dtype=torch.long),
                [
                    self.images_files[idx],
                    self.target_files_hipp[idx],
                    self.target_files_baga[idx],
                    self.target_files_extra[idx],
                ],
            )


class LISAInferenceDataset(Dataset):
    def __init__(
        self,
        patch_size=(128, 128, 128),
        use_atlas=False,  # TODO: fix and create atlas for inference on the fly
        interpolate=True,
    ):
        self.use_atlas = use_atlas
        self.patch_size = patch_size
        self.interpolate = interpolate

        files = dutil.load_data_paths_inference(use_atlas)
        if self.use_atlas:
            images_files, atlas_files = files
        else:
            images_files = files
            atlas_files = None
        self.images_files = images_files

        self.subjects = [
            tio.Subject(
                image=tio.ScalarImage(path=str(self.images_files[idx])),
                atlas=tio.LabelMap(path=str(atlas_files[idx])) if self.use_atlas else None,
            )
            for idx in range(len(self.images_files))
        ]

        # Load all images into RAM
        # First load moves the data to RAM, afterwards it is cached
        for subject in self.subjects:
            subject["image"].load()
            if use_atlas:
                subject["atlas"].load()
        self.set_tranforms()

    def set_tranforms(self):
        """Set transforms for the dataset."""
        standard_transforms = dutil.get_standard_transforms(patch_size=self.patch_size, interpolate=self.interpolate)

        self.transform = tio.Compose(standard_transforms)

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, idx):

        subject = self.subjects[idx]
        original_spatial_shape = subject["image"].data.shape[1:]

        subject_tr = self.transform(subject)
        # from cache in RAM
        img_tensor = subject_tr["image"].data

        original_image_tensor = subject["image"].data
        if self.use_atlas:
            atlas_tensor = subject_tr["atlas"].data
            atlas_tensor = atlas_tensor.squeeze().long()
            return (
                img_tensor,
                original_image_tensor,
                atlas_tensor,
                torch.tensor(original_spatial_shape, dtype=torch.long),
                self.images_files[idx],
            )
        return (
            img_tensor,
            original_image_tensor,
            torch.tensor(original_spatial_shape, dtype=torch.long),
            self.images_files[idx],
        )
