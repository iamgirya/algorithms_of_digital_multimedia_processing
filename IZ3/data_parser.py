import gzip
import shutil


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


import dicom2nifti


# Folder "CT-0" consist of CT scans having normal lung tissue,
# no CT-signs of viral pneumonia.

# for x in os.listdir(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies"):
#     dicom2nifti.convert_directory(os.path.join(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies", x,
#                                                    os.listdir(os.path.join(
#                                                        r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies",
#                                                        x))[0]), os.path.join(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies", x))
#     unpack_single_gzip_in_folder(os.path.join(r"C:\Users\RedMa\OneDrive\Рабочий стол\data\bad\700_800_studies", x))
