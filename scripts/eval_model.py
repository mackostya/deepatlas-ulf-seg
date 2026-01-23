import os
import torch

import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src import utils
from src.losses import RelativeVolmeError


from src.loop import TrainingLoop, FinetuningLoop
from src import visualizations as vis
from dataloaders.dataset import LISAEvalDataset
from monai.metrics import HausdorffDistanceMetric as HausdorffDistanceMonai
from monai.metrics import DiceMetric as DiceMetricMonai
from monai.metrics import SurfaceDistanceMetric as SurfaceDistanceMonai


def create_table(
    dice_monai_score,
    hausdorff_score,
    hausdorff_95_score,
    assd_score,
    relative_error,
    name="Subject Scores",
    record=False,
):
    rows = np.array(
        [
            ["Hip Left"],
            ["Hip Right"],
            ["E Vent Left"],
            ["E Vent Roght"],
            ["BG Caud Left"],
            ["BG Caud Right"],
            ["BG Lent Left"],
            ["BG Lent Left"],
        ]
    )
    columns = ["", "Dice", "Huasdorf", "Huasdorf 95", "ASSD", "Relative Volume Error"]
    table = Table(title=name)

    # If Average Scores, add average and std
    if name == "Average Scores":
        rows = np.concatenate([rows, [["Average"]]], axis=0)
        rows = np.concatenate([rows, [["Average 2a"]]], axis=0)
        rows = np.concatenate([rows, [["Average 2b"]]], axis=0)

        selected_indices_2a = [0, 1]
        dice_avg_2a = dice_monai_score[selected_indices_2a].mean(dim=0, keepdim=True)
        dice_std_2a = dice_monai_score[selected_indices_2a].std(dim=0, keepdim=True)
        hausdorff_avg_2a = hausdorff_score[selected_indices_2a].mean(dim=0, keepdim=True)
        hausdorff_std_2a = hausdorff_score[selected_indices_2a].std(dim=0, keepdim=True)
        hausdorff_95_avg_2a = hausdorff_95_score[selected_indices_2a].mean(dim=0, keepdim=True)
        hausdorff_95_std_2a = hausdorff_95_score[selected_indices_2a].std(dim=0, keepdim=True)
        assd_avg_2a = assd_score[selected_indices_2a].mean(dim=0, keepdim=True)
        assd_std_2a = assd_score[selected_indices_2a].std(dim=0, keepdim=True)
        relative_error_avg_2a = relative_error[selected_indices_2a].mean(dim=0, keepdim=True)
        relative_error_std_2a = relative_error[selected_indices_2a].std(dim=0, keepdim=True)

        selected_indices_2b = [4, 5, 6, 7]
        dice_avg_2b = dice_monai_score[selected_indices_2b].mean(dim=0, keepdim=True)
        dice_std_2b = dice_monai_score[selected_indices_2b].std(dim=0, keepdim=True)
        hausdorff_avg_2b = hausdorff_score[selected_indices_2b].mean(dim=0, keepdim=True)
        hausdorff_std_2b = hausdorff_score[selected_indices_2b].std(dim=0, keepdim=True)
        hausdorff_95_avg_2b = hausdorff_95_score[selected_indices_2b].mean(dim=0, keepdim=True)
        hausdorff_95_std_2b = hausdorff_95_score[selected_indices_2b].std(dim=0, keepdim=True)
        assd_avg_2b = assd_score[selected_indices_2b].mean(dim=0, keepdim=True)
        assd_std_2b = assd_score[selected_indices_2b].std(dim=0, keepdim=True)
        relative_error_avg_2b = relative_error[selected_indices_2b].mean(dim=0, keepdim=True)
        relative_error_std_2b = relative_error[selected_indices_2b].std(dim=0, keepdim=True)

        dice_avg = dice_monai_score.mean(dim=0, keepdim=True)
        dice_std = dice_monai_score.std(dim=0, keepdim=True)
        hausdorff_avg = hausdorff_score.mean(dim=0, keepdim=True)
        hausdorff_std = hausdorff_score.std(dim=0, keepdim=True)
        hausdorff_95_avg = hausdorff_95_score.mean(dim=0, keepdim=True)
        hausdorff_95_std = hausdorff_95_score.std(dim=0, keepdim=True)
        assd_avg = assd_score.mean(dim=0, keepdim=True)
        assd_std = assd_score.std(dim=0, keepdim=True)
        relative_error_avg = relative_error.mean(dim=0, keepdim=True)
        relative_error_std = relative_error.std(dim=0, keepdim=True)

        dice_monai_score = torch.cat([dice_monai_score, dice_avg, dice_avg_2a, dice_avg_2b], dim=0)
        dice_monai_std = torch.cat(
            [torch.zeros_like(dice_monai_score[:-3]), dice_std, dice_std_2a, dice_std_2b], dim=0
        )
        hausdorff_score = torch.cat([hausdorff_score, hausdorff_avg, hausdorff_avg_2a, hausdorff_avg_2b], dim=0)
        hausdorff_std = torch.cat(
            [torch.zeros_like(hausdorff_score[:-3]), hausdorff_std, hausdorff_std_2a, hausdorff_std_2b], dim=0
        )
        hausdorff_95_score = torch.cat(
            [hausdorff_95_score, hausdorff_95_avg, hausdorff_95_avg_2a, hausdorff_95_avg_2b], dim=0
        )
        hausdorff_95_std = torch.cat(
            [torch.zeros_like(hausdorff_95_score[:-3]), hausdorff_95_std, hausdorff_95_std_2a, hausdorff_95_std_2b],
            dim=0,
        )
        assd_score = torch.cat([assd_score, assd_avg, assd_avg_2a, assd_avg_2b], dim=0)
        assd_std = torch.cat([torch.zeros_like(assd_score[:-3]), assd_std, assd_std_2a, assd_std_2b], dim=0)
        relative_error = torch.cat(
            [relative_error, relative_error_avg, relative_error_avg_2a, relative_error_avg_2b], dim=0
        )
        relative_error_std = torch.cat(
            [torch.zeros_like(relative_error[:-3]), relative_error_std, relative_error_std_2a, relative_error_std_2b],
            dim=0,
        )
    else:
        dice_monai_std = torch.zeros_like(dice_monai_score)
        hausdorff_std = torch.zeros_like(hausdorff_score)
        hausdorff_95_std = torch.zeros_like(hausdorff_95_score)
        assd_std = torch.zeros_like(assd_score)
        relative_error_std = torch.zeros_like(relative_error)

    for column in columns:
        table.add_column(column)

    # Format: value +- std (only for averages, otherwise just value)
    rows_with_scores = []
    for i in range(len(rows)):
        dice_val = dice_monai_score[i].item()
        dice_std_val = dice_monai_std[i].item()
        hausdorff_val = hausdorff_score[i].item()
        hausdorff_std_val = hausdorff_std[i].item()
        hausdorff_95_val = hausdorff_95_score[i].item()
        hausdorff_95_std_val = hausdorff_95_std[i].item()
        assd_val = assd_score[i].item()
        assd_std_val = assd_std[i].item()
        relerr_val = relative_error[i].item()
        relerr_std_val = relative_error_std[i].item()

        def fmt(val, std):
            if std > 0:
                return f"{val:.2f} ± {std:.2f}"
            else:
                return f"{val:.2f}"

        row = [
            rows[i][0],
            fmt(dice_val, dice_std_val),
            fmt(hausdorff_val, hausdorff_std_val),
            fmt(hausdorff_95_val, hausdorff_95_std_val),
            fmt(assd_val, assd_std_val),
            fmt(relerr_val, relerr_std_val),
        ]
        rows_with_scores.append(row)

    for row in rows_with_scores:
        table.add_row(*row, style="bright_green")

    console = Console(record=record)
    return console, table


if __name__ == "__main__":

    config_id = 0  # specifies which model configuration to use
    cfg = utils.load_cfg(config_id)
    utils.initialize_random_seed(cfg["random_seed"])
    checkpoint_path = "PATH_TO_YOUR_CHECKPOINT"  # path to .ckpt file with trained model
    save_nii_files = False
    use_atlas = "atlas" in cfg["model_type"]
    dataset = LISAEvalDataset(
        patch_size=cfg["patch_size"], split="val", use_atlas=use_atlas, interpolate=cfg["interpolate"]
    )  # (128, 128, 128)

    val_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )
    print("Validation set size:", len(dataset))

    # load the model
    loop = TrainingLoop.load_from_checkpoint(
        checkpoint_path,
        model_type=cfg["model_type"],
        class_weights=cfg["class_weights"],
        map_location="cuda",
    )

    loop.eval()
    loop.freeze()
    loop.to("cuda")

    dice_monai = DiceMetricMonai(
        include_background=False,
        reduction="none",
        get_not_nans=False,
    )
    hausdorff_monai = HausdorffDistanceMonai(
        include_background=False,
    )
    hausdorff_monai_95 = HausdorffDistanceMonai(
        include_background=False,
        percentile=95,
    )
    surface_distance_monai = SurfaceDistanceMonai(
        include_background=False,
        distance_metric="euclidean",
        reduction="none",
        get_not_nans=False,
    )
    relative_vol_error = RelativeVolmeError(include_background=False)
    results_folder = "outputs/eval_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    dice_scores = []
    hausdorff_scores = []
    hausdorff_95_scores = []
    surface_distance_scores = []
    relative_vol_errors = []
    img_files_list = []
    hipp_labels_list = []
    basal_labels_list = []
    extra_labels_list = []
    pred_files_names = []
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if use_atlas:
                volume, original_volume, original_label, atlas, original_shape, files = batch
            else:
                volume, original_volume, original_label, original_shape, files = batch
                atlas = None
            img_file, hipp_file, basal_file, extra_file = files

            ################################### process image and label ########################################
            # The prediction is made on interpolated volume, so we need to reinterpolate the predictions back.
            # The original volumes are in RAS order (D, H, W), but the model expects SAR order (W, H, D)
            # The dataset transforms the original volumes to SAR order, so we need to swap axes back
            # to RAS order before saving the predictions.
            ####################################################################################################

            b, n, d, h, w = original_label.shape
            original_label = original_label.detach().cpu().long()  # (1, 9, W, H, D)
            original_shape = original_shape.float().numpy()  # [D, H, W]

            volume = volume.to(loop.device)
            if use_atlas:
                atlas = atlas.to(loop.device)
            pred = loop(volume, atlas=atlas).detach().cpu()  # output shape is (1, 9, W_c, H_c, D_c)

            pred = pred.argmax(dim=1)
            pred = F.one_hot(pred.long(), num_classes=9).movedim(-1, 1).float()
            pred = pred.swapaxes(2, -1)  # SAR -> RAS (1, 9, W_c, H_c, D_c) -> (1, 9, D_c, H_c, W_c)
            if cfg["interpolate"]:
                pred_inv = torch.nn.functional.interpolate(
                    pred,
                    size=(
                        int(original_shape[0, 0]),
                        int(original_shape[0, 1]),
                        int(original_shape[0, 2]),
                    ),
                    mode="nearest",
                )  # pred_inv shape is (1, 9, W, H, D)
            else:
                pred_inv = utils.inverse_cropping(pred, original_shape)  # inverse cropping to original shape

            # pred_inv = pred_inv.swapaxes(2, -1)  # SAR -> RAS (1, 9, W, H, D) -> (1, 9, D, H, W)
            # (Optional, if proper one hot encoding) fix invalid one-hot predictions of background
            # pred_inv = utils.clean_one_hot_background(pred_inv)

            dice_monai_score = dice_monai(pred_inv, original_label).mean(dim=0)  # average across the batch

            hausdorff_score = hausdorff_monai(pred_inv, original_label).mean(dim=0)  # average across the batch

            hausdorff_95_score = hausdorff_monai_95(pred_inv, original_label).mean(dim=0)  # average across the batch

            assd_score = surface_distance_monai(pred_inv, original_label).mean(dim=0)  # average across the batch

            relative_error = relative_vol_error(pred=pred_inv, true=original_label).mean(
                dim=0
            )  # average across the batch

            if save_nii_files == True:
                filename = f"{results_folder}/pred_{i}.nii.gz"
                utils.save_nii_gz(pred_inv, filename)

                pred_files_names.append(filename)
                img_files_list.append(img_file[0])
                hipp_labels_list.append(hipp_file[0])
                basal_labels_list.append(basal_file[0])
                extra_labels_list.append(extra_file[0])

            console, table = create_table(
                dice_monai_score,
                hausdorff_score,
                hausdorff_95_score,
                assd_score,
                relative_error,
                name=f"Volume {i} Scores",
            )
            console.print(table)

            if not dice_monai_score.isnan().any():
                dice_scores.append(dice_monai_score)
                hausdorff_scores.append(hausdorff_score)
                hausdorff_95_scores.append(hausdorff_95_score)
                surface_distance_scores.append(assd_score)
                relative_vol_errors.append(relative_error)

            vis.save_segmentation_for_all_types(
                original_volume.detach().cpu(),
                original_label.detach().cpu(),
                pred_inv.detach().cpu(),
                results_path=f"{results_folder}/segmentation_{i}_" + "{seg_type}.png",
            )
        if save_nii_files:
            pd.DataFrame(
                {
                    "original_file": img_files_list,
                    "pred_files": pred_files_names,
                    "labels_hipp": hipp_labels_list,
                    "labels_basal": basal_labels_list,
                    "labels_extra": extra_labels_list,
                }
            ).to_csv(f"{results_folder}/files_list.csv", index=False)

    print(f"-------------------------------------- Average --------------------------------------------")
    dice_scores = torch.stack(dice_scores, dim=0)
    hausdorff_scores = torch.stack(hausdorff_scores, dim=0)
    hausdorff_95_scores = torch.stack(hausdorff_95_scores, dim=0)
    surface_distance_scores = torch.stack(surface_distance_scores, dim=0)
    relative_vol_errors = torch.stack(relative_vol_errors, dim=0)

    console, table = create_table(
        dice_scores.mean(dim=0),
        hausdorff_scores.mean(dim=0),
        hausdorff_95_scores.mean(dim=0),
        surface_distance_scores.mean(dim=0),
        relative_vol_errors.mean(dim=0),
        name="Average Scores",
        record=True,
    )

    console.print(table)
    console.save_text(f"{results_folder}/average_scores.txt")
