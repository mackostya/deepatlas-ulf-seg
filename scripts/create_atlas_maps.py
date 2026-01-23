from dataloaders import data_utils as dutils
from src.atlas_utils import create_atlas_from_volumes

if __name__ == "__main__":
    visualize = True
    data_path = dutils.initialize_data_path()
    print(f"Data path: {data_path}")
    subject_img_paths, _, _, _ = dutils.load_all_file_paths(data_path)
    create_atlas_from_volumes(subject_img_paths)
