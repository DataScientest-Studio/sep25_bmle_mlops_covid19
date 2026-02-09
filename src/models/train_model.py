from datetime import datetime
import os
from pathlib import Path
import shutil

from src.models.models_to_test import EfficientNetv2B0_model_augmented
from src.utils.data_utils import build_masked_dataset_by_classes, import_dataset
from src.utils.image_utils import ensure_dirs, oversample_train_class_1, split_dataset

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATASET_DIR_NAME = "COVID-19_Radiography_Dataset"
DATASET_REP = DATA_DIR / DATASET_DIR_NAME
STRUCTURED_DATASET = DATA_DIR / "structured_dataset_for_training"
TMP_DIR = DATA_DIR / "tmp"
MODELS_DIR = BASE_DIR / "src" / "models"


def prepare_dataset(force: bool = False) -> None:
    if not os.path.isdir(DATASET_REP) or force:
        import_dataset(DATASET_DIR_NAME, DATASET_REP)

    if force and STRUCTURED_DATASET.exists():
        shutil.rmtree(STRUCTURED_DATASET)

    if not os.path.isdir(STRUCTURED_DATASET):
        build_masked_dataset_by_classes(DATASET_REP, TMP_DIR)
        ensure_dirs(output_dir=STRUCTURED_DATASET)
        split_dataset(source_dir=TMP_DIR, output_dir=STRUCTURED_DATASET, train_ratio=0.8)
        oversample_train_class_1(output_dir=STRUCTURED_DATASET, oversample_multiplier=4.0)
        shutil.rmtree(TMP_DIR)


def train_and_save(epochs: int = 200, force: bool = False, model_name: str | None = None) -> Path:
    prepare_dataset(force=force)

    if not model_name:
        now = datetime.now().strftime("%y%m-%m-%d-%H-%M-%S")
        model_name = f"EfficientNetv2B0_model_trained_{now}"

    model = EfficientNetv2B0_model_augmented(
        data_folder=STRUCTURED_DATASET,
        save_model_folder=MODELS_DIR,
        model_name=model_name,
        batch_size=16,
        big_dataset=True,
        oversampling=True,
    )

    model.load_data()
    model.fit(epochs=epochs)
    model.save()
    model.evaluate()

    return MODELS_DIR / f"{model_name}.keras"


if __name__ == "__main__":
    model_path = train_and_save()
    print(f"Model trained and saved successfully: {model_path}")
