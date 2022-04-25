'''
    DatasetCleanup.py - this script is used for cleaning up unlabeled images from disk. Dataset .csv file is loaded 
                        and each image that is not in the .csv file will be deleted from disk. Work of this script 
                        cannot be undone!
'''

import os 
import glob 
import numpy as np
from datetime import datetime

imgExstension = '.jpg'
datasetDirectory = '../cnn/dataset'
labelsPath = '../cnn/dataset/dataset.csv'

# read input .csv file
def read_csv(filepath):
    result = []
    dat_file = open(filepath,'r', encoding="utf-8-sig")
    lines=dat_file.readlines()
    for line in lines:
        if len(line)>0:
            p1 = line.find(',')
            filename = line[0:p1]
            categ = line[p1+1:]
            s = filename+','+categ
            result.append(s)
    return result

# Load filenames and labels
def load_filenames_and_labels(imgs_dir, label_path):

    filenames = []
    labels = []
    lines = read_csv(label_path)
    lines.pop(0) # remove header

    for line in lines:

        if ((len(line) > 0)):
            p1 = line.find(',')
            fname = line[0:p1]
            p1 = p1+1
            filenames.append(os.path.join(imgs_dir, fname))

            cat=line[p1:]

            cat = cat.rstrip(',\n')
            cat = cat.split(',')

            cnt_cat = 0
            for item in cat:
                cat[cnt_cat] = float(item)
                cnt_cat = cnt_cat + 1
            cat = np.asarray(cat)

            labels.append(cat)
    
    return [labels, filenames]

def datasetImages(datasetDirectory, imgExstension):

    imgsOnDisk = []

    for file in os.listdir(datasetDirectory):
        if file.endswith(imgExstension):
            imgsOnDisk.append(os.path.join(datasetDirectory, file))

    return imgsOnDisk

def compareDatasetWithDiskImgsAndDelete(datasetFilenames, diskFilenames):
    
    cnt = 0
    for fname in diskFilenames:
        if fname not in datasetFilenames:
            cnt += 1
            os.remove(os.path.abspath(fname))

    print("Deleted " + str(cnt) + " images from the disk.")

if __name__ == "__main__":
    script_start = datetime.now()

    # load dataset .csv 
    [labels, filenames] = load_filenames_and_labels(datasetDirectory, labelsPath)

    # read list of images from disk in dataset directory
    imgsOnDisk = datasetImages(datasetDirectory, imgExstension)

    # compare dataset filenames with list read from the dataset directory
    # permanently delete unlabeled images
    compareDatasetWithDiskImgsAndDelete(filenames, imgsOnDisk)

    script_end = datetime.now()
    print(script_end - script_start)