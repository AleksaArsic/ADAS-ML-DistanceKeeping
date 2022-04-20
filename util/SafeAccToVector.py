'''
    SafeAccToVector.py - this script will take values of safeAcc from dataset and based on the individual values produce vector 
                         of three values [safeAcc notSafeAcc keepDistance]
'''

import os
import shutil
from datetime import datetime
from pathlib import Path

inCsvPath = '..\cnn\dataset\dataset.csv'
outCsvPath = '..\cnn\dataset'

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

def safeAccToVector(data):

    data[0].append('keepDistance')
    data[0].append('notSafeAcc')

    # iterate trough data from .csv file
    for i in range(1,len(data)):
        # take safeAcc parameter
        safeAcc = float(data[i][-1])
        tempData = []

        # convert safeAcc to [safeAcc keepDistance notSafeAcc]
        if (safeAcc == 0):
            tempData = [1, 0, 0]
        elif (safeAcc == 0.5):
            tempData = [0, 1, 0]
        elif (safeAcc == 1):
            tempData = [0, 0, 1]

        data[i][-1] = tempData[0]
        data[i].append(tempData[1])
        data[i].append(tempData[2])

    return data

def saveCSVFile(outPath, data):
    # Format: imgName, Vpresent, Vpl, Vpr, safeToAcc

    outFilePath = os.path.join(outPath, "datasetSafeAccVector_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".csv")

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

if __name__ == "__main__":

    # load .csv file
    data = parseCSVFile(inCsvPath)
    
    # process data
    outData = safeAccToVector(data)

    print(outData[0])

    # save new .csv file
    saveCSVFile(outCsvPath, outData)