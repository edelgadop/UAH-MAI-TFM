import tarfile
import logging
import re
import os
import numpy as np
import pandas as pd
import cv2


def extract_images(files: list, output_dir):
    for file in files:
        logging.info(f"Extracting contents of {file} ...")
        with tarfile.open(file) as tfd:
            tfd.extractall(output_dir)
            logging.info(f"Finished extracting contents of {file}")


def is_numeric(s: str) -> bool:
    m = re.match(pattern="\d+", string=s)
    if m is not None:
        return True
    else:
        return False


def get_input_tensor(dir1: str, metadata: pd.DataFrame, img_size=64, max_dirs=25) -> tuple:
    images_tensors = []
    gender_labels = []
    age_labels = []
    ids = []
    for subdir in os.listdir(dir1):
        if is_numeric(subdir) and int(subdir) <= max_dirs:
            print(f"Analyzing subdir {subdir} / {max_dirs} ...")
            images_paths = os.listdir(dir1 + subdir)
            for img_path in images_paths:
                data_img = metadata.loc[metadata["id"]==img_path,:]
                if img_path.lower().endswith(".jpg") and data_img["valid"].values[0]:
                    img = cv2.imread(dir1 + subdir + "/" + img_path, 0)  # monochrome
                    if (img.shape[0] >= img_size) and (img.shape[1] >= img_size):
                        img_array = cv2.resize(img, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
                        img_array = img_array / 255 # normalization
                        img_tensor = np.expand_dims(img_array, axis=0)
                        images_tensors.append(img_tensor)
                        gender_labels.append(data_img["gender"].values[0])
                        age_labels.append(data_img["age"].values[0])
                        ids.append(data_img["id"].values[0])
    print("Building input tensor and target arrays ... ")
    return np.concatenate(images_tensors), np.array(gender_labels), np.array(age_labels), np.array(ids)
