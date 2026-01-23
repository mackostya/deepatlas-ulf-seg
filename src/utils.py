import os
import yaml
import sys
import torch
import datetime
import SimpleITK as sitk
import torch.nn.functional as F
import numpy as np
import platform
import torchio as tio


def load_cfg(config_id: int, path: str = "configs/config_training.yml"):
    config = yaml.safe_load(open(path))
    return config[config_id]


def load_nii_gz(path):
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)
    return img


# Function to initialize the random seed
def initialize_random_seed(seed_value):
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def create_log_dir(model_type: str, additional_training_comment: str = "") -> str:
    iso_date = datetime.datetime.today()

    log_dir = (
        os.path.abspath(os.curdir)
        + f"/logs/{model_type}/{iso_date.year}_{iso_date.month}_{iso_date.day}/{iso_date.hour}-{iso_date.minute:0>2}_{additional_training_comment}/"
    )

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return log_dir


def save_nii_gz(tensor: torch.Tensor, filename: str):
    """
    Save a tensor as a NIfTI file.
    Args:
        tensor: A tensor of shape (1, C, D, H, W) where C is the number of channels.
        path: The directory where the NIfTI file will be saved.
    """
    tensor_index = tensor.argmax(dim=1, keepdim=True).squeeze(0)  # from (1, C, D, H, W) to (1, D, H, W)
    image = tio.ScalarImage(tensor=tensor_index)
    image.save(filename)


def init_system(model_type: str, additional_training_comment: str = "", config_id: int = 0) -> str:
    # Initialize the seed
    try:
        config_id = int(sys.argv[1])
    except IndexError:
        config_id = config_id
    cfg = load_cfg(config_id)

    initialize_random_seed(cfg["random_seed"])
    if platform.system() == "Windows":
        log_dir = create_log_dir(model_type, additional_training_comment)
    elif platform.system() == "Linux":
        try:
            log_dir = os.environ["LOGFOLDER_PATH"] + f"/{additional_training_comment}_{config_id}/"
        except KeyError:
            log_dir = create_log_dir(model_type, additional_training_comment)
    else:
        raise NotImplementedError("Unsupported platform for logging directory initialization.")

    return log_dir, cfg


def inverse_cropping(tensor: torch.Tensor, original_shapes: torch.Tensor) -> torch.Tensor:
    """
    Undo CropOrPad on a batch of volume-tensors.
    tensor: (B, C, Dp, Hp, Wp)
    original_shapes: (B, 3) giving (Do, Ho, Wo) per sample
    returns: (B, C, Do, Ho, Wo)
    """
    restored = []
    for i in range(tensor.shape[0]):
        vol = tensor[i]
        orig = tuple(int(x) for x in original_shapes[i])
        restored.append(_invert_crop_one(vol, orig))
    return torch.stack(restored, dim=0)


def _invert_crop_one(vol: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
    """
    vol: (C, Dp, Hp, Wp)
    orig_shape: (Do, Ho, Wo)
    """
    C, Dp, Hp, Wp = vol.shape
    Do, Ho, Wo = orig_shape

    def slices(patch, orig):
        if patch > orig:
            pad_before = (patch - orig) // 2
            return pad_before, pad_before + orig, 0, orig
        else:
            crop_before = (orig - patch) // 2
            return 0, patch, crop_before, crop_before + patch

    d_ss, d_se, d_ds, d_de = slices(Dp, Do)
    h_ss, h_se, h_ds, h_de = slices(Hp, Ho)
    w_ss, w_se, w_ds, w_de = slices(Wp, Wo)

    out = torch.zeros((C, Do, Ho, Wo), device=vol.device, dtype=vol.dtype)
    out[:, d_ds:d_de, h_ds:h_de, w_ds:w_de] = vol[:, d_ss:d_se, h_ss:h_se, w_ss:w_se]
    return out


def clean_one_hot_background(pred_inv):
    """
    Fix invalid one-hot predictions:
    - Remove background where any other class is predicted.
    - Assign background where nothing is predicted.

    Args:
        pred_inv: torch.Tensor of shape (1, C, D, H, W), binary one-hot encoded

    Returns:
        torch.Tensor of same shape, cleaned.
    """
    pred_inv = pred_inv.clone()  # avoid modifying original

    # Remove background (channel 0) where any foreground class is predicted
    has_foreground = pred_inv[:, 1:, ...].sum(dim=1, keepdim=True) > 0  # shape: (1, 1, D, H, W)
    pred_inv[:, 0][has_foreground.squeeze(1)] = 0

    # Assign background (channel 0) where no class is predicted at all
    nothing_predicted = pred_inv.sum(dim=1, keepdim=True) == 0  # shape: (1, 1, D, H, W)
    pred_inv[:, 0][nothing_predicted.squeeze(1)] = 1

    return pred_inv
