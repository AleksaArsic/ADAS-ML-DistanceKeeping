'''
    RenameImgs.py - script used for fast reaneming of large amount of images. 
                    Names will be enumerated with consecutive numbers.
'''
import os
from datetime import datetime

startEnumeration = 71400 # from which number to start enumeration of new img names
imgExstension = '.jpg'   # image extension to look for on disk
imgPath = '..\camera_sensors_output\center_town06_addition2_3' # path to folder containing images to rename
leadingZeros = 6 # indicates how many leading zeros to add to new image names


def renameImages(imgPath):
    
    names = [f for f in os.listdir(imgPath) if os.path.isfile(os.path.join(imgPath, f))]

    indexesNotImgs = []

    for i in range(len(names)):
        fileExstension = '.' + names[i].split('.')[-1]
        
        if(fileExstension != imgExstension):
            indexesNotImgs.append(i)

    for i in range(len(indexesNotImgs) - 1, -1, -1):
        names.pop(indexesNotImgs[i])

    newNames = (list(range(startEnumeration, startEnumeration + len(names))))

    for i in range(len(names)):
        # rename image on disk
        imgName = str(newNames[i]).zfill(leadingZeros) + imgExstension
        os.rename(os.path.join(imgPath, names[i]), os.path.join(imgPath, imgName))


if __name__ == "__main__":
    script_start = datetime.now()

    # rename images 
    renameImages(imgPath)

    script_end = datetime.now()
    print(script_end - script_start)