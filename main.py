from pathlib import Path
from src.utils.data_utils import organize_custom_dataset


if __name__ == "__main__":
    
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    DATASET_DIR_PATH = DATA_DIR / "structured_dataset"
    ORIGINAL_DATASET_PATH = DATA_DIR / "COVID-19_Radiography_Dataset_init"
    print(DATA_DIR)
    params = {
                    "folderA": 300 // 3,
                    "folderB": 300 // 3,
                    "folderC": 300 // 3,
                    "folderD": 300 
                }
    organize_custom_dataset(dataset_path=str(ORIGINAL_DATASET_PATH), output_root=str(DATASET_DIR_PATH), images_by_folder=params, replace=True)