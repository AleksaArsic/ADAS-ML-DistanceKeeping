'''
    AugmentationMirror.py - this script is used for mirroring images (flip horizontally) with criteria that 
                            the parameter safeToAcc from Dataset is 0.5
                            Note: mirror augmented images have prefix '_mirror_'
'''

import os
from jupyterlab_server import LabServerApp 
import numpy as np
from datetime import datetime
from PIL import Image, ImageOps

safeToAccCriteria = 0.5
datasetPath = '../cnn/dataset'
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

def flipImagesAndSave(labels, filenames):

    # found all images with safeAcc criteria
    imgsToFlip = []
    labelsToFlip = []
    namesOfFlippedImgs = []

    for i in range(len(labels)):
        if(labels[i][-1] == safeToAccCriteria):
            imgsToFlip.append(filenames[i])
            labelsToFlip.append(labels[i])

    # flip all images that satisfy the criteria
    cnt = 0
    for fname in imgsToFlip:
        img = Image.open(os.path.abspath(fname))
        img = img.convert('RGB')
        imgFlipped = ImageOps.mirror(img)

        name = '_mirror_' + os.path.abspath(fname).split('\\')[-1]
        namesOfFlippedImgs.append(name)

        imgFlipped.save(os.path.join(os.path.abspath(datasetPath), name))
        cnt += 1

    # flip labels if needed 
    for i in range(len(labelsToFlip)):
        tempRight = labelsToFlip[i][2]
        labelsToFlip[i][2] = labelsToFlip[i][1]
        labelsToFlip[i][1] = tempRight

    # append flipped labels to dataset .csv
    labelsToCsv = []

    for i in range(len(imgsToFlip)):
        labelsToCsv.append([namesOfFlippedImgs[i]] + list(labelsToFlip[i]))

    with open(labelsPath, "a") as file:

        for i in range(len(labelsToCsv)):
            line = ''
            for j in range(len(labelsToCsv[i])):
                line += (str(labelsToCsv[i][j]) + ',')

            line += '\n'

            file.write(line)

if __name__ == "__main__":

    script_start = datetime.now()

    # read dataset .csv file
    [labels, filenames] = load_filenames_and_labels(datasetPath, labelsPath)

    # flip and save new flipped images
    flipImagesAndSave(labels, filenames)

    script_end = datetime.now()
    print(script_end - script_start)