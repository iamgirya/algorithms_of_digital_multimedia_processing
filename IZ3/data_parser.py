import gzip
import os
import pathlib
import shutil
import dicom2nifti
import xlrd
import csv


def unpack_single_gzip_in_folder(folder_path):
    files = os.listdir(folder_path)

    gz_files = [file for file in files if file.endswith(".gz")]

    if len(gz_files) == 1:
        gz_file_path = os.path.join(folder_path, gz_files[0])

        output_file_path = os.path.splitext(gz_file_path)[0]

        with gzip.open(gz_file_path, "rb") as f_in, open(
            output_file_path, "wb"
        ) as f_out:
            shutil.copyfileobj(f_in, f_out)

        os.remove(gz_file_path)
        print(f"Распаковано: {gz_file_path} -> {output_file_path}")
    else:
        print("Ошибка: Не удалось определить единственный файл .gz в указанной папке.")


# Folder "CT-0" consist of CT scans having normal lung tissue,
# no CT-signs of viral pneumonia.
pack_name = "100_200_studies"
path = (
    r"dataset\MosMedData-CT-HEMORRHAGE-type VIII\\"
    + pack_name
    + "\\"
    + pack_name
    + "\\"
)
files = os.listdir(path)
xlsx_files = [file for file in files if file.endswith(".xlxs")]
if len(xlsx_files) == 1:
    xlxs_file = xlsx_files[0]

    def csv_from_excel():
        wb = xlrd.open_workbook(path + xlxs_file)
        sh = wb.sheet_by_name("Clinical params")
        your_csv_file = open("labels.csv", "w")
        wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)
        for rownum in range(sh.nrows):
            wr.writerow(sh.row_values(rownum))
        your_csv_file.close()

    csv_from_excel()

    labels = {}

    with open(
        str(rel_path(path + "labels.csv")), newline="", encoding="utf-8"
    ) as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="'")
        for row in reader:
            if row[0][0] == "s":
                continue
            labels[row[0]] = row[2] == "normal"

# runs the csv_from_excel function:
csv_from_excel()

for x in os.listdir(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies"):
    dicom2nifti.convert_directory(
        os.path.join(
            r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies",
            x,
            os.listdir(
                os.path.join(
                    r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies", x
                )
            )[0],
        ),
        os.path.join(
            r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies", x
        ),
    )
    unpack_single_gzip_in_folder(
        os.path.join(
            r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies", x
        )
    )
