from datetime import datetime
import os
from pathlib import Path
import shutil

from src.models.models_to_test import EfficientNetv2B0_model_augmented
from src.utils.data_utils import build_masked_dataset_by_classes, import_dataset
from src.utils.image_utils import ensure_dirs, oversample_train_class_1, split_dataset
from src.utils.db_dataset_utils import fetch_image_rows_sync, download_dataset_from_rows

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATASET_DIR_NAME = "COVID-19_Radiography_Dataset"
DATASET_REP = DATA_DIR / DATASET_DIR_NAME
STRUCTURED_DATASET = DATA_DIR / "structured_dataset_for_training"
TMP_DIR = DATA_DIR / "tmp"
DB_RAW_DIR = DATA_DIR / "db_raw"
DB_STRUCTURED_DATASET = DATA_DIR / "db_structured_dataset"
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


def prepare_db_dataset(
    force: bool = True,
    table: str = "images_dataset",
    limit: int | None = None,
    batch_size: int = 1000,
    apply_masks: bool = True,
) -> None:
    if force and DB_STRUCTURED_DATASET.exists():
        shutil.rmtree(DB_STRUCTURED_DATASET)

    if force and DB_RAW_DIR.exists():
        shutil.rmtree(DB_RAW_DIR)

    if not DB_STRUCTURED_DATASET.exists():
        rows = fetch_image_rows_sync(table=table, limit=limit, batch_size=batch_size)
        if not rows:
            raise RuntimeError("No images found in database.")

        saved = download_dataset_from_rows(
            rows=rows,
            output_dir=DB_RAW_DIR,
            apply_masks=apply_masks,
            replace=True,
        )
        if saved == 0:
            raise RuntimeError("No images downloaded from database.")

        ensure_dirs(output_dir=DB_STRUCTURED_DATASET)
        split_dataset(source_dir=DB_RAW_DIR, output_dir=DB_STRUCTURED_DATASET, train_ratio=0.8)
        oversample_train_class_1(output_dir=DB_STRUCTURED_DATASET, oversample_multiplier=4.0)
        shutil.rmtree(DB_RAW_DIR)


def train_and_save(
    epochs: int = 200,
    force: bool = False,
    model_name: str | None = None,
    data_source: str = "db",
    db_table: str = "images_dataset",
    db_limit: int | None = None,
    db_batch_size: int = 1000,
    apply_masks: bool = True,
) -> Path:
    if data_source == "db":
        prepare_db_dataset(
            force=force,
            table=db_table,
            limit=db_limit,
            batch_size=db_batch_size,
            apply_masks=apply_masks,
        )
        data_folder = DB_STRUCTURED_DATASET
    else:
        prepare_dataset(force=force)
        data_folder = STRUCTURED_DATASET

    if not model_name:
        now = datetime.now().strftime("%y%m-%m-%d-%H-%M-%S")
        model_name = f"EfficientNetv2B0_model_trained_{now}"

    model = EfficientNetv2B0_model_augmented(
        data_folder=data_folder,
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
