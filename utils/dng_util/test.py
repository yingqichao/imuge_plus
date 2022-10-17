import glob
import os
print(os.getcwd())
from .DNG import DNG


def extract_info(dataset_root):
    dng_files = glob.glob(dataset_root+'*/photos/*.dng')
    for i in dng_files:
        print(i)
    # dng = DNG(emptyDNG)
    # dng.openDNG()
    # dng.readHeader()
    # dng.readIFDs()


if __name__ == '__main__':
    dataset_root = '/ssd/fivek_total/fivek_dataset/raw_photos/'
    extract_info(dataset_root)