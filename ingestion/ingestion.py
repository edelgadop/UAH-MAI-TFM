import os
import pandas as pd
import scipy.io
import numpy as np


def get_metadata_from_mat_file(matfile: str, key: str) -> list:
    metadata = []
    mat = scipy.io.loadmat(matfile)

    _id = mat[key][0][0][2][0]
    gender = mat[key][0][0][3][0]
    dob = np.int32(mat[key][0][0][0][0] / 365)
    photo = mat[key][0][0][1][0]
    age = photo - dob

    for i in range(len(_id)):
        record = {
            "id": str(_id[i][0][3:]),
            "year_of_birth": dob[i],
            "year_of_photo": photo[i],
            "age": age[i],
            "gender": np.uint8(gender[i]),
            "valid": (age[i] <= 100) & (age[i] > 0)
        }
        metadata.append(record)
    return metadata


def get_metadata(matfile_imdb: str, matfile_wiki: str) -> pd.DataFrame:
    if matfile_imdb is None and matfile_wiki is None:
        keys = []
        files = []
    elif matfile_imdb is None:
        keys = ["wiki"]
        files = [matfile_wiki]
    elif matfile_wiki is None:
        keys = ["imdb"]
        files = [matfile_imdb]
    else:
        keys = ["imdb", "wiki"]
        files = [matfile_imdb, matfile_wiki]

    metadata = [get_metadata_from_mat_file(files[i], keys[i]) for i in range(len(keys))]

    return pd.DataFrame([item for sublist in metadata for item in sublist])