import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from nilearn import image
from matplotlib.patches import Patch


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def show_segmentation_image(image, seg, pred):

    # constructing image
    fig, ax = plt.subplots(1, figsize=(8, 8))

    ax.imshow(image, cmap="gray")
    ax.imshow(np.ma.masked_where(seg == 0, seg), cmap="jet", alpha=0.5)
    ax.imshow(np.ma.masked_where(pred == 0, pred), cmap="rainbow", alpha=0.5)
    # Get the color from the colormap for value 1 (since mask is 0/1)
    true_color = plt.cm.jet(Normalize()(1))
    pred_color = plt.cm.rainbow(Normalize()(1))

    # If you want to add extra_mask, use a different color, e.g. plt.cm.spring(Normalize()(1))
    legend_patches = [
        mpatches.Patch(color=true_color, label="True"),
        mpatches.Patch(color=pred_color, label="Predicted"),
    ]

    ax.set_title(f"Original Image with Segmentation Overlays")
    ax.legend(handles=legend_patches, loc="lower right", fontsize="large", frameon=True)
    ax.axis("off")
    return fig


def vis_segmentation_volume_per_type(img, labels, preds, seg_type, apply_sigmoid=True):

    image = img.squeeze()  # H, W
    if apply_sigmoid:
        preds = _sigmoid(preds)  # C, H, W
        preds = np.where(preds > 0.5, 1, 0)
    if seg_type == "hipp":
        labels = labels[1] + labels[2]
        preds = preds[1] + preds[2]
    elif seg_type == "basal":
        labels = labels[5] + labels[6] + labels[7] + labels[8]
        preds = preds[5] + preds[6] + preds[7] + preds[8]
    elif seg_type == "extra":
        labels = labels[3] + labels[4]
        preds = preds[3] + preds[4]
    else:
        raise ValueError("Unknown segmentation type")
    labels = labels.clip(0, 1)
    preds = preds.clip(0, 1)
    fig = show_segmentation_image(image, labels, preds)
    return fig


def save_segmentation_for_all_types(original_volume, original_label, pred, results_path=None):

    if results_path is None:
        ValueError("results_path must be provided to save segmentation images.")

    slice_id = 60  # empirical choice for hippocampus, can be changed

    fig = vis_segmentation_volume_per_type(
        original_volume[0, :, :, :, slice_id],
        original_label[0, :, :, :, slice_id],
        pred[0, :, :, :, slice_id],
        seg_type="hipp",  # "basal", "extra"
        apply_sigmoid=False,
    )

    fig.savefig(results_path.format(seg_type="hipp"))

    slice_id = 90  # empirical choice for basal ganglia, can be changed

    fig = vis_segmentation_volume_per_type(
        original_volume[0, :, :, :, slice_id],
        original_label[0, :, :, :, slice_id],
        pred[0, :, :, :, slice_id],
        seg_type="basal",  # "basal", "extra"
        apply_sigmoid=False,
    )

    fig.savefig(results_path.format(seg_type="basal"))

    fig = vis_segmentation_volume_per_type(
        original_volume[0, :, :, :, slice_id],
        original_label[0, :, :, :, slice_id],
        pred[0, :, :, :, slice_id],
        seg_type="extra",  # "basal", "extra"
        apply_sigmoid=False,
    )

    fig.savefig(results_path.format(seg_type="extra"))
    plt.close("all")  # close all figures to free memory


def plot_middle_slice(img_array, title, slice_id=None, output_file=None):
    slice_id = img_array.shape[2] // 2 if slice_id is None else slice_id
    plt.imshow(img_array[:, :, slice_id], cmap="gray")
    plt.title(title)
    plt.axis("off")
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()


def plot_labeled_overlay(bg_img, atlas_img, labels, output_file, slice_index=None, title=""):
    # # Resample atlas to background image
    if bg_img.shape != atlas_img.shape:
        atlas_img = image.resample_to_img(atlas_img, bg_img, interpolation="nearest")

    # Extract data arrays
    bg_data = bg_img.get_fdata()
    atlas_data = atlas_img.get_fdata()

    if slice_index is None:
        slice_index = bg_data.shape[2] // 2

    bg_slice = bg_data[:, :, slice_index].T  # transpose for correct orientation
    atlas_slice = atlas_data[:, :, slice_index].T

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(bg_slice, cmap="gray", origin="lower")
    im = ax.imshow(np.ma.masked_where(atlas_slice == 0, atlas_slice), cmap="jet", alpha=0.4, origin="lower")

    # Prepare legend handles and labels
    unique_regions = np.unique(atlas_slice)
    handles = []
    legend_labels = []
    cmap = plt.get_cmap("jet")

    # Remove background (0) from unique regions for coloring/legend
    unique_regions = unique_regions[unique_regions != 0]
    print("Unique regions in atlas slice (excluding background):", unique_regions)

    # Map region values to consecutive color indices
    region_to_color_idx = {region: idx for idx, region in enumerate(unique_regions)}
    n_regions = len(region_to_color_idx)

    for region_value in unique_regions:
        if region_value == 0:
            continue
        label_idx = int(region_value)
        if label_idx < len(labels):
            label = labels[label_idx]
            color = cmap(region_value / np.max(unique_regions))
            handles.append(Patch(facecolor=color, edgecolor="black"))
            legend_labels.append(label)

    # Add legend to the plot
    if handles:
        ax.legend(
            handles, legend_labels, loc="upper right", fontsize=9, title="Regions", title_fontsize=10, frameon=True
        )

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
