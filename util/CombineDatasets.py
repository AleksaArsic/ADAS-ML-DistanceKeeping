## CombineDatasets.py - Script that used in unification of data from multiple iterations of dataset 
## collection process into one .csv file
#####################CAUTION#####################CAUTION#####################CAUTION#####################CAUTION###########################
## !!! CAUTION: This script will copy all images to desired location without removing old ones, sufficient disk space must be provided  !!!
## !!! CAUTION: This script assumes images are in the same folder as .csv files                                                         !!!
## !!! CAUTION: This script will rename all images to prevent possible conflict of image names                                          !!!
#####################CAUTION#####################CAUTION#####################CAUTION#####################CAUTION###########################

import os
import shutil
from datetime import datetime
from pathlib import Path

errCode = -1 # error code
noError = 0  # OK

leadingZeros = 6 # indicates how many leading zeros to add to new image names

imgNameId = 0   # imgName index in both .csv files
throttleId = 1  # throttle index in cnn .csv file
breakId = 2     # break index in cnn .csv file 

# inputh paths to .csv files to be concatenated
inPaths = [ '../camera_sensors_output/center/out.csv',                                      \
            '../camera_sensors_output/center_town01/UnifyDataset_03_04_2022_20_20_07.csv',  \
            '../camera_sensors_output/center_town02/UnifyDataset_03_04_2022_20_21_19.csv' ]

outPath = '../cnn/dataset/'   # path to output folder of combined dataset

def parseCSVFile(csvPath):
    data = []
    
    csv = open(Path(csvPath), "r") 
    csvFile = csv.readlines()

    print("Loading .csv file", csvPath)

    for line in csvFile:
        if(len(line) > 0):
            data.append(line.replace('\n', '').split(','))

    csv.close()

    return data

def saveCSVFile(outPath, data):
    # Format: imgName, Vpresent, Vpl, Vpr, safeToAcc

    outFilePath = os.path.join(outPath, "CombineDataset_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".csv")

    csvFile = open(outFilePath, "w+") 

    print("Saving .csv file to location", outFilePath)

    for i in range(len(data)):
        line = ''

        for j in range(len(data[i])):
            line += str(data[i][j]) + ','

        line += '\n'

        csvFile.write(line)

    csvFile.close()

    print("Done.")

def combineCSVFiles(csvFiles):
    
    outCSVFile = []
    noOfImages = []

    for i in range(len(csvFiles)):
        
        # delete header files from all but first .csv
        if (i > 0):
            csvFiles[i].pop(0)

        outCSVFile += csvFiles[i]
        noOfImages.append(len(csvFiles[i]))

    return outCSVFile, noOfImages

def copyImages(inPaths, outPath, combinedData, noOfImages):

    # exit due to possible loss of data if directory already exists
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    else:
        return errCode

    index = 0

    for i in range(len(inPaths)):
        # number of images in one folder
        cnt = noOfImages[i] + index

        # base input path
        baseInPath = inPaths[i].split('/')
        baseInPath.pop(-1)

        # construct input path without .csv file in it
        strInPath = ''

        for k in range(len(baseInPath)):
            strInPath += (baseInPath[k] + '/')

        for j in range(index, cnt):

            # skip first because of the .csv header
            if(j == 0):
                continue

            inImgPath = os.path.join(strInPath, combinedData[j][0])
            outImgPath = os.path.join(outPath, combinedData[j][0])

            # copy images from inImgPath to outImgPath
            shutil.copy(inImgPath, outImgPath) 

        index += noOfImages[i]


def renameImages(imgPath, combinedData):
    
    newNames = (list(range(len(combinedData))))

    imgExstension = '.' + combinedData[1][0].split('.')[-1]

    for i in range(1, len(combinedData)):
        # rename image on disk
        imgName = str(newNames[i]).zfill(leadingZeros) + imgExstension
        os.rename(os.path.join(imgPath, combinedData[i][0]), os.path.join(imgPath, imgName))

        # rename the img in .csv
        combinedData[i][0] = imgName

    # return combinedData with new img names
    return combinedData

def main():

    csvFiles = []

    # parse .csv files
    for i in range(len(inPaths)):
        csvFiles.append(parseCSVFile(inPaths[i]))

    # combine all .csv files to one
    combinedData, noOfImages = combineCSVFiles(csvFiles)

    # copy images to new destination
    retVal = copyImages(inPaths, outPath, combinedData, noOfImages)

    if(retVal == errCode):
        print("ERROR! " + str(outPath) + " exists, possible loss of data!")
        return errCode

    # rename newly copied images
    combinedData = renameImages(outPath, combinedData)

    # save .csv file
    saveCSVFile(outPath, combinedData)

if __name__ == "__main__":

    script_start = datetime.now()

    main()

    script_end = datetime.now()
    print (script_end - script_start)