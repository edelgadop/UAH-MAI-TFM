import tarfile
import logging
import re
import os
import numpy as np
import matplotlib.pyplot as plt
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


def get_input_tensor(dir1: str) -> np.ndarray:
    images_tensors = []
    for subdir in os.listdir(dir1):
        if is_numeric(subdir):
            print(f"Analyzing subdir {subdir} / 99 ...")
            images_paths = os.listdir(dir1 + subdir)
            for img_path in images_paths:
                if img_path.lower().endswith(".jpg"):
                    img = plt.imread(dir1 + subdir + "/" + img_path)
                    if (img.shape[0] >= 100) and (img.shape[1] >= 100) and (len(img.shape) == 3):
                        img_array = cv2.resize(img, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
                        img_array = img_array / 255  # normalization
                        img_tensor = np.expand_dims(img_array, axis=0)
                        images_tensors.append(img_tensor)
    print("Building input tensor ... ")
    return np.concatenate(images_tensors)