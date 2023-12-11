import gzip
import os
import pathlib
import shutil
import dicom2nifti
import pandas as pd
import csv


def unpack_single_gzip_in_folder(folder_path, dicom_file, is_good):
    files = os.listdir(folder_path)

    gz_files = [file for file in files if file.endswith(".gz")]

    if len(gz_files) == 1:
        gz_file_path = os.path.join(folder_path, gz_files[0])

        sub_folder = "good\\" if is_good == 1 else "bad\\"
        output_file_path = "parsed_data\\" + sub_folder + dicom_file + ".nii"

        with gzip.open(gz_file_path, "rb") as f_in, open(
            output_file_path, "wb"
        ) as f_out:
            shutil.copyfileobj(f_in, f_out)

        os.remove(gz_file_path)
        print(f"Распаковано: {gz_file_path} -> {output_file_path}")
    else:
        print("Ошибка: Не удалось определить единственный файл .gz в указанной папке.")


def parse_some_pack(pack_name):
    path = (
        r"dataset\MosMedData-CT-HEMORRHAGE-type VIII\\"
        + pack_name
        + "\\"
        + pack_name
        + "\\"
    )
    files = os.listdir(path)
    xlsx_files = [file for file in files if file.endswith(".xlsx")]
    if len(xlsx_files) >= 1:
        xlxs_file = xlsx_files[0]

        read_file = pd.read_excel(path + xlxs_file)
        read_file.to_csv(path + "labels.csv", index=None, header=False, sep="!")

        labels = {}

        with open(path + "labels.csv", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter="!", quotechar="'")
            for row in reader:
                if row[0][0] == "s":
                    continue
                labels[row[0]] = row[2] == "normal"

        for dicom_file in labels.keys():
            dicom2nifti.convert_directory(
                os.path.join(
                    path,
                    dicom_file,
                    os.listdir(
                        os.path.join(
                            path,
                            dicom_file,
                        )
                    )[0],
                ),
                os.path.join(
                    path,
                    dicom_file,
                ),
            )
            unpack_single_gzip_in_folder(
                os.path.join(
                    path,
                    dicom_file,
                ),
                dicom_file,
                labels[dicom_file],
            )


packs_name = [
    # "0_100_studies",  # смешанные
    # "100_200_studies", # здоровые # загружен
    "200_300_studies",  # здоровые
    # "300_400_studies",  # здоровые
    # "400_500_studies",  # здоровые
    # "500_600_studies",  # больные # загружен
    "600_700_studies",  # больные
    # "700_800_studies",  # больные
]

for pack in packs_name:
    parse_some_pack(pack)
