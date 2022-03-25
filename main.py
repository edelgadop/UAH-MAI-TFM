from preprocessing.preprocessing import extract_images


IMG_DIR = "../images"
FILE_IMDB = "../imdb_crop.tar"
FILE_WIKI = "../wiki_crop.tar"

if __name__ == "__main__":
    extract_images([FILE_IMDB, FILE_WIKI], IMG_DIR)