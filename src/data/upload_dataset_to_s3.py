import os
import csv
import boto3
from botocore.client import Config
from datetime import datetime

def upload_dataset_and_generate_csv(bucket_name, access_key, secret_key, local_root, s3_prefix="", output_csv="dataset.csv"):
    """
    Upload images and masks to Backblaze B2 S3 bucket and generate a CSV with URLs.

    CSV format:
    image_url, mask_url, img_class, now, now
    """
    class_type = "0"
    
    now = datetime.now()
    
    # Connexion S3 compatible Backblaze
    s3 = boto3.client(
        "s3",
        endpoint_url="https://s3.eu-central-003.backblazeb2.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4')
    )

    csv_rows = []

    for dossier in os.listdir(local_root):
        if dossier == "COVID":
            class_type = "1"
        else:
            class_type = "0"
            
        dossier_path = os.path.join(local_root, dossier)
        if not os.path.isdir(dossier_path):
            continue

        images_folder = os.path.join(dossier_path, "images")
        masks_folder = os.path.join(dossier_path, "masks")

        if not os.path.exists(images_folder) or not os.path.exists(masks_folder):
            continue

        # On assume que les noms des fichiers images et masks correspondent
        for filename in os.listdir(images_folder):

            if not filename.lower().endswith(".png"):
                continue

            image_path = os.path.join(images_folder, filename)
            mask_path = os.path.join(masks_folder, filename)

            if not os.path.isfile(mask_path):
                print(f"Warning: mask missing for {filename}")
                continue

            # Définir les clés S3
            image_s3_key = os.path.join(s3_prefix, dossier, "images", filename).replace("\\", "/")
            mask_s3_key = os.path.join(s3_prefix, dossier, "masks", filename).replace("\\", "/")

            try:
                # Upload
                s3.upload_file(image_path, bucket_name, image_s3_key)
                s3.upload_file(mask_path, bucket_name, mask_s3_key)
                pass
            except Exception as e:
                print(e)

            # Générer les URLs publiques
            image_url = f"https://{bucket_name}.s3.eu-central-003.backblazeb2.com/{image_s3_key}"
            mask_url = f"https://{bucket_name}.s3.eu-central-003.backblazeb2.com/{mask_s3_key}"

            csv_rows.append([image_url, mask_url, class_type, now, now])
            print(f"Uploaded {filename} and added to CSV")

    # Écriture du CSV
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_url", "mask_url", "class_type", "injection_date", "created_at"])
        writer.writerows(csv_rows)

    print(f"CSV file saved to {output_csv}")

# Exemple d'utilisation
if __name__ == "__main__":
    upload_dataset_and_generate_csv(
        bucket_name="mlops-liora-covid19",
        access_key="00315d4f4eeb9d40000000001",
        secret_key="K003cHU2uuu3kAJ+A/X0k60g6LoU4vs",
        local_root="./data/COVID-19_Radiography_Dataset",
        s3_prefix="dataset",
        output_csv="dataset.csv"
    )
