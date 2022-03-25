import tarfile
import logging
import os
import pandas as pd


def extract_images(files: list, output_dir):
    for file in files:
        logging.info(f"Extracting contents of {file} ...")
        with tarfile.open(file) as tfd:
            tfd.extractall(output_dir)
            logging.info(f"Finished extracting contents of {file}")

def load_images(path: str) -> pd.DataFrame:


def image2array():