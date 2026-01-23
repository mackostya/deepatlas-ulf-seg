# Atlas-Augmented Semantic Segmentation for Robust Ultra-Low-Field Pediatric Brain Imaging

This repository contains the official PyTorch implementation of our approach for **Task 2a (bilateral hippocampus segmentation)** and **Task 2b (bilateral basal ganglia segmentation)** of the [LISA 2025 Challenge](https://www.synapse.org/Synapse:syn65670170/wiki/631438) on ultra-low-field pediatric brain MRI.

---

## Abstract

Low-field MRI offers a portable, cost-effective alternative to conventional high-field scanners but suffers from reduced signal-to-noise ratio and spatial inhomogeneity, which compromise the accuracy and consistency of automated brain structure segmentation. In this work, we introduce atlas-augmented deep learning models that integrate probabilistic anatomical priors to enhance the delineation of pediatric hippocampus and basal ganglia in ultra-low-field MRI (0.064 T).  
We evaluate seven pipelines on the LISA 2025 dataset (79 T2-weighted scans): baseline VNet, nnU-Net, and MedSAM2 variants (2D and 3D decoders), as well as atlas-augmented VNet, atlas-augmented nnU-Net, and atlas-augmented MedSAM2-3D. For VNet and MedSAM2-3D, probabilistic maps from the Pauli and Harvard-Oxford atlases are encoded and fused with intermediate feature maps, while nnU-Net ingests priors as additional input channels. Baseline nnU-Net attains mean DSCs of 0.71 for hippocampus and 0.86 for basal ganglia; atlas augmentation yields modest hippocampal gains (HD95 $\downarrow$0.05, ASSD $\downarrow$0.06) and more pronounced improvements in basal ganglia segmentation, reflecting richer prior information for larger structures. VNet and MedSAM2 variants exhibit limited hippocampal benefit, highlighting the strength of nnU-Net's adaptive framework. Our findings establish atlas-augmented nnU-Net as a new benchmark for robust segmentation in resource-constrained, low-field imaging environments.

## Requirements and Installation

Clone the repository and create the conda environment:

```bash
conda env create -f environment.yml
conda activate lisa25
```

Install the MedSAM2 dependency separately:

```bash
pip install --no-deps git+https://github.com/bowang-lab/MedSAM2.git@main#egg=medsam2
```

---

## Dataset Setup

The repository expects the official LISA 2025 dataset to be organized as follows:

```
/YOUR_PATH/lisa2025/data_organized/
└── Task 2 - Segmentation
    ├── Low Field Images
    ├── Subtask 2a - Hippocampus Segmentations
    ├── Subtask 2b - Basal Ganglia Segmentations
    └── Extra Segmentations/Ventricle
```

Training/validation splits are defined using CSV files:

- `configs/train_data.csv`
- `configs/val_data.csv`

Expected CSV format:

| images | target_hipp | target_baga | target_extra | atlas (optional) |
|--------|-------------|-------------|---------------|------------------|
| LISA_0001_ciso.nii.gz | LISA_0001_HF_hipp.nii.gz | LISA_0001_HF_baga.nii.gz | LISA_0001_vent.nii.gz | LISA_0001_atlas.nii.gz |

Only filenames are stored in CSV; full paths are constructed internally based on the dataset root.

### ⚠️ IMPORTANT: Set your own dataset path

Before running training or evaluation, you need to update the dataset paths manually in:
```
dataloaders/data_utils.py
```

This file contains not filled paths like:
```
    YOUR_DATA_PATH_HERE
    YOUR_ATLAS_DATA_PATH
```

Modify ALL paths in data_utils.py so they point to your own local directories that contain:  
    - LISA 2025 Task 2 training images and labels  
    - Atlas volumes (required only for atlas-based models)

---

## Model Configurations

Model selection is controlled by numeric `config_id` entries in `configs/config_training.yml`. Available configurations include:

| Config ID | Model Type            | Batch Size | Patch Size        | Atlas Priors | Class Weights | Notes |
|------------|----------------------|-------------|-------------------|---------------|----------------|--------|
| 0          | vnet                 | 3           | 128×128×128       | No            | No             | Baseline VNet |
| 1          | medsam2              | 8           | 128×128×128       | No            | No             | Baseline MedSAM2 |
| 2          | medsam2_3d          | 2           | 128×128×128       | No            | No             | MedSAM2 + 3D decoder |
| 3          | medsam2_3d_atlas    | 2           | 128×128×128       | Yes           | No             | Atlas fusion |
| 4          | vnet                 | 3           | 128×128×128       | No            | Yes            | Class-balanced VNet |
| 5          | medsam2              | 8           | 128×128×128       | No            | Yes            | Class-balanced MedSAM2 |
| 6          | medsam2_3d          | 2           | 128×128×128       | No            | Yes            | Class-balanced 3D decoder |
| 7          | medsam2_3d_atlas    | 2           | 128×128×128       | Yes           | Yes            | Atlas + class-weighted |
| 10         | medsam2_3d          | 1           | 256×256×128       | No            | No             | Large field-of-view |
| 11         | medsam2_3d_atlas    | 1           | 256×256×128       | Yes           | No             | Large FOV + atlas |
| 12         | vnet_atlas          | 2           | 128×128×128       | Yes           | No             | VNet + atlas |
---

## Training

Open `scripts/train.py` and set:
- `config_id = YOUR_SELECTED_ID` at the top

Then run:
```bash
python scripts/train.py
```
Training logs and checkpoints are saved automatically under `logs/`.

---

## Atlas Priors (Optional)

Atlas priors enhance anatomical coherence for low SNR images. To prepare atlas priors:

```bash
python scripts/create_atlas_maps.py
python scripts/combine_atlas.py
```

Generated atlas volumes must correspond to the `atlas` field in dataset CSVs.

---

## Evaluation

Open `scripts/eval_model.py` and set:
- `config_id = YOUR_SELECTED_ID`
- `checkpoint_path = "PATH_TO_YOUR_CHECKPOINT.ckpt"`

Then run:
```bash
python scripts/eval_model.py
```

Metrics computed:
- Dice Coefficient
- Hausdorff Distance
- Average Symmetric Surface Distance (ASSD)
- Relative Volume Error

Outputs are saved in:
```
outputs/eval_results/
```

---

## Pretrained Weights

MedSAM2 encoder checkpoints are available in `checkpoints/` and can be downloaded using:

```bash
sh download_medsam.sh
```

---

## nnU-Net with Atlas Priors 

This repository also provides optional support for training **nnU-Net v2**.  
Unlike standard nnU-Net, this implementation uses **3 input channels**:

| Channel | Description |
|----------|-------------|
| `0000.nii.gz` | Low-field MRI (T2-weighted) |
| `0001.nii.gz` | Pauli subcortical atlas prior |
| `0002.nii.gz` | Harvard–Oxford cortical atlas prior |

The atlas priors are already supported in this repository and can be generated using the provided scripts.

### 1: Prepare atlas volumes and change data paths
If atlas files are not yet created, run:
```
python scripts/create_atlas_maps.py
python scripts/combine_atlas.py
```
These scripts generate atlas priors used during conversion and training.   
Before converting data, update dataset and atlas root paths in:
```
dataloaders/data_utils.py
```
Replace the hardcoded paths (dataset root and atlas folders) with your own local path.

### 2: Install nnU-Net v2
```
Install nnU-Net v2
pip install nnunetv2
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```
`setup_nnunet_env.sh` can be used as an example environment script.

### 3: Convert dataset to nnU-Net format (with atlas channels)

```
python scripts/convert_lisa_to_nnUNet.py
```
This will create a new nnU-Net dataset in `nnUNet_raw/Dataset001_LISA/` containing 3-channel inputs: MRI + Pauli atlas + HO atlas.

### 4: Train nnU-Net
Train a single fold:
```
nnUNetv2_train 1 3d_fullres 0
```
Train all folds:
```
for FOLD in 0 1 2 3 4; do nnUNetv2_train 1 3d_fullres $FOLD; done
```

### 5: Prepare inference data
```
python scripts/convert_lisa_to_nnUNet_inference.py
```
This converts input images + atlas priors for prediction.

### 6: Run inference
```
nnUNetv2_predict -i nnUNet_inference_files -o nnUNet_predictions -d 1 -c 3d_fullres -f all
```

---


## Citation

If you use this repository in your research, please cite our challenge submission:

```
@inproceedings{lavronenko_atlas-augmented_2026,
	title = {Atlas-Augmented Semantic Segmentation for Robust Ultra-Low-Field Pediatric Brain Imaging},
	doi = {10.1007/978-3-032-14417-1_8},
	booktitle = {Low Field Pediatric Brain Magnetic Resonance Image Segmentation and Quality Assurance},
	publisher = {Springer Nature Switzerland},
	author = {Lavronenko, Kostiantyn and Yilmaz, Rueveyda and Chen, Zhu and Stegmaier, Johannes and Schulz, Volkmar},
	editor = {Lepore, Natasha and Linguraru, Marius George},
	year = {2026},
}
```

Additionally cite foundational works:

```
@misc{milletari2016vnet,
      title={V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation}, 
      author={Fausto Milletari and Nassir Navab and Seyed-Ahmad Ahmadi},
      year={2016},
      eprint={1606.04797},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1606.04797}, 
}

@article{MedSAM2,
    title={MedSAM2: Segment Anything in 3D Medical Images and Videos},
    author={Ma, Jun and Yang, Zongxin and Kim, Sumin and Chen, Bihui and Baharoon, Mohammed and Fallahpour, Adibvafa and Asakereh, Reza and Lyu, Hongwei and Wang, Bo},
    journal={arXiv preprint arXiv:2504.03600},
    year={2025}
}
```

---

## Contact

For questions, please contact:   
📧 Kostiantyn.Lavronenko@lfb.rwth-aachen.de  
📧 Rueveyda.Yilmaz@lfb.rwth-aachen.de  
📧 Zhu.Chen@lfb.rwth-aachen.de