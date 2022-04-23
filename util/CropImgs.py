'''
    CropImgs.py - script used for cropping images to desired dimensions and save to desired location.
                  Script will crop ROI for the purposes of ADAS-Distance-Keeping project.
                  It can easly be modifed for cropping desired ROI.
'''
import os
import sys 
from PIL import Image
from datetime import datetime

errCode = -1 # error code
noError = 0  # OK

imgExstension = '.jpg'   # image extension to look for on disk

inPath = '../camera_sensors_output/center_town03_addition2_3'
outPath = './CropImgs_out'

targetImgWidht = 600
targetImgHeight = 370

# cuts Images to target width and height
def cutImage(pilImg, targetW, targetH):

    imgWidth, imgHeight = pilImg.size

    widthCrop = imgWidth - targetW

    leftPoint = widthCrop / 2
    upperPoint = imgHeight - targetH
    rightPoint = imgWidth - leftPoint
    lowerPoint = imgHeight

    pilImg = pilImg.crop((leftPoint, upperPoint, rightPoint, lowerPoint))

    return pilImg

def loadImages(inPath):
    # exit due to possible loss of data if directory already exists
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    else:
        print("ERROR! " + str(outPath) + " exists, possible loss of data!")
        sys.exit()

    print("loading images...")

    imgs = []
    filenames = []

    for file in os.listdir(inPath):
        if file.endswith(imgExstension):
            filenames.append(file)
            imgPath = os.path.join(inPath, file)
            img = Image.open(imgPath)
            imgs.append(img)

    return imgs, filenames

def saveImages(outPath, filenames, imgs):

    print("saving images...")

    for i in range(len(filenames)):
        imgOutPath = os.path.join(outPath, filenames[i])
        imgs[i].convert('RGB').save(imgOutPath)

if __name__ == '__main__':
    script_start = datetime.now()

    # load images
    imgs, filenames = loadImages(inPath)

    # crop images
    for i in range(len(imgs)):
        imgs[i] = cutImage(imgs[i], targetImgWidht, targetImgHeight)

    # save images
    saveImages(outPath, filenames, imgs)

    script_end = datetime.now()
    print(script_end - script_start)