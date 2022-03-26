import tarfile
import logging
import re
import os
import numpy as np
import matplotlib.pyplot as plt
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


def get_input_tensor(dir1: str, metadata: pd.DataFrame) -> tuple:
    images_tensors = []
    gender_labels = []
    age_labels = []
    ids = []
    for subdir in os.listdir(dir1):
        if is_numeric(subdir):
            print(f"Analyzing subdir {subdir} / 99 ...")
            images_paths = os.listdir(dir1 + subdir)
            for img_path in images_paths:
                data_img = metadata.loc[metadata["id"] == img_path,:]
                if img_path.lower().endswith(".jpg") and data_img["valid"].values[0]:
                    img = plt.imread(dir1 + subdir + "/" + img_path)
                    if (img.shape[0] >= 100) and (img.shape[1] >= 100) and (len(img.shape) == 3):
                        img_array = cv2.resize(img, dsize=(250,250), interpolation=cv2.INTER_CUBIC)
                        img_array = img_array / 255  # normalization
                        img_tensor = np.expand_dims(img_array, axis=0)
                        images_tensors.append(img_tensor)
                        gender_labels.append(data_img["gender"].values[0])
                        age_labels.append(data_img["age"].values[0])
                        ids.append(data_img["id"].values[0])
    print("Building input tensor and target arrays ... ")
    return np.concatenate(images_tensors), np.array(gender_labels), np.array(age_labels), np.array(ids)
