'''
    PIDLongCtrlGsAnalysis.py - script used for analysis of data taken from Grid Search algorithm with purpose 
                               of finding optimal PID Longitudinal Controller parameters (Kp, Ki, Kd)
'''

from fileinput import filename
import os
from datetime import datetime
from pathlib import Path

import numpy as np

directory = '../pid_grid_search/' # directory with grid search .csv files

def parseCSVFile(csvPath):
    data = []
    
    csv = open(Path(csvPath), "r") 
    csvFile = csv.readlines()

    print("Loading .csv file", csvPath)

    for line in csvFile:
        if(len(line) > 0):
            data.append(line.replace('\n', '').split(','))

    # remove header from data
    data.pop(0)

    csv.close()

    return data

def parseData(data):
    # parse data to two arrays:
    # [throttle], [brake]

    # transpose data
    transposedData = np.transpose(np.array(data))

    return [transposedData[0], transposedData[1]]

if __name__ == "__main__":
    script_start = datetime.now()

    # get all .csv filenames from directory
    filenames = []

    for file in os.listdir(directory):
        if file.endswith(".csv"):
            filenames.append(os.path.join(directory, file))

    print(filenames)

    for i in range(len(filenames)):
        # parse each .csv file
        data = parseCSVFile('../pid_grid_search/MainSimulation_05_24_2022_20_52_55.csv')

        # sort data into two arrays:
        # [throttle], [brake]
        transposedData = np.transpose(np.array(data))

        throttle, brake = transposedData[0], transposedData[1]

        print(throttle)
        print(brake)

        # TO-DO: process/analyse data

        # save analysis results

        print(data)
    
    script_end = datetime.now()
    print(script_end - script_start)