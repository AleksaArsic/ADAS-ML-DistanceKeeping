## CombineDatasets.py - Script that used in unification of data from multiple iterations of dataset 
## collection process into one .csv file

import os
from datetime import datetime
from pathlib import Path

errCode = -1 # error code

imgNameId = 0   # imgName index in both .csv files
throttleId = 1  # throttle index in cnn .csv file
breakId = 2     # break index in cnn .csv file 

yolov3OutCSVPath = '../camera_sensors_output/center_town02/out_yolov3.csv' # paths to input .csv files
cnnOutCSVPath = '../camera_sensors_output/center_town02/out_cnn.csv'       # paths to input .csv files
outCSVPath = '../camera_sensors_output/center_town02'                      # path to output .csv folder

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

def combineCSVFiles():
    pass

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

def main():
    # parse YOLOv3 .csv file
    yolov3Data = parseCSVFile(yolov3OutCSVPath)

    # parse CNN .csv file
    cnnData = parseCSVFile(cnnOutCSVPath)

    # combine all .csv files to one
    combinedData = combineCSVFiles()

    # save .csv file
    saveCSVFile(outCSVPath, combinedData)

if __name__ == "__main__":
    main()