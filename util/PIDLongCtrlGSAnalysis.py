'''
    PIDLongCtrlGsAnalysis.py - script used for analysis of data taken from Grid Search algorithm with purpose 
                               of finding optimal PID Longitudinal Controller parameters (Kp, Ki, Kd)
'''

import os
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
from sqlalchemy import func

directory = '../pid_grid_search_3/' # directory with grid search .csv files

minGoodnessThreshold = -0.7
maxGoodnessThreshold = 0.7

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

    return [np.float32(transposedData[0]), np.float(transposedData[1])]

def combineThrottleBrake(throttle, brake):

    fun = []

    for i in range(len(throttle)):
        if (float(throttle[i]) > 0):
            fun.append(float(throttle[i]))
        else:
            fun.append(float(brake[i]))

    return fun

def generateTime(timeStep, timeLen):

    time = []

    time.append(0)

    for i in range(1, timeLen):
        time.append(time[i - 1] + timeStep)

    return time

def generateIndividualTime(timeStep, throttle, brake):

    throttleLen = len(throttle)
    brakeLen = len(brake)

    timeLen = 0

    if (throttleLen > brakeLen):
        timeLen = throttleLen
    else:
        timeLen = brakeLen

    time = generateTime(timeStep, timeLen)

    return time

def calculateGoodness(function, min, max):

    cntMin = 0
    cntMax = 0
    cntNonZero = 0

    for i in range(len(function)):
        if(function[i] < 0):
            cntNonZero += 1
            if(function[i] > min):
                cntMin += 1
        elif(function[i] > 0):
            cntNonZero += 1
            if(function[i] < max):
                cntMax += 1

    return ((0.5 * cntMin) + (0.5 * cntMax)) / cntNonZero

# plot analysis data 
def plotAnalysis(throttle, brake, throttleBrake, longestTime, time, outputDir):

    plt.clf()

    #Plottig the throttle and brake functions
    plt.figure(figsize=(30, 10))
    plt.plot(longestTime[:len(throttle)], throttle, 'b', label='Throttle function')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title('Throttle analysis in time')
    plt.xlabel('Time')
    plt.ylabel('Thorttle')
    plt.legend()
    plt.savefig(os.path.join(outputDir, 'throttle.png'))

    plt.clf()

    plt.figure(figsize=(30, 10))
    plt.plot(longestTime[:len(brake)], brake, 'b', label='Brake function')
    #plt.axhline(y=0, color='black', linestyle='-')
    plt.title('Brake analysis in time')
    plt.xlabel('Time')
    plt.ylabel('Brake')
    plt.legend()
    plt.savefig(os.path.join(outputDir, 'brake.png'))

    plt.clf()

    plt.figure(figsize=(30, 10))
    plt.plot(time, throttleBrake, 'b')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title('Throttle/Brake analysis in time')
    plt.xlabel('Time')
    plt.ylabel('Throttle/Brake')
    plt.savefig(os.path.join(outputDir, 'throttleBrakeFunction.png'))

if __name__ == "__main__":
    script_start = datetime.now()

    # get all .csv filenames from directory
    filenames = []

    for file in os.listdir(directory):
        if file.endswith(".csv"):
            filenames.append(os.path.join(directory, file))

    # Analyse data for each Grid Search iteration
    bestThrottle = []
    bestBrake = []
    bestThrottleBrakeFun = []
    goodness = []
    for i in range(len(filenames)):

        # parse each .csv file
        data = parseCSVFile(filenames[i])

        # sort data into two arrays:
        # [throttle], [brake]
        transposedData = np.transpose(np.array(data))

        throttle, brake = transposedData[0], transposedData[1]

        throttle = [float(i) for i in throttle]
        brake = [float(i) for i in brake]

        # combine throttle and brake in one function
        throttleBrakeFun = combineThrottleBrake(throttle, brake)

        # calculate how good is throttleBrakeFun
        goodness.append(calculateGoodness(throttleBrakeFun, minGoodnessThreshold, maxGoodnessThreshold))

        # save best throttle, brake, throttleBrakeFun
        bestThrottle.append(throttle)
        bestBrake.append(brake)
        bestThrottleBrakeFun.append(throttleBrakeFun)

        if(goodness[-1] >= 0.20):

            outputFolder = './PIDLongCtrlGSAnalysis_output_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + "_" + str(goodness[-1])

            # create output folder
            if not os.path.exists(outputFolder):
                os.makedirs(outputFolder)
            else:
                print("Folder with the name " + str(outputFolder) + " already exists. Possible loss of data.")
                print("Aborting...")
                sys.exit()

            # find best function
            #goodness.sort(reverse=True)
            #bestGoodness = max(goodness)
            #idx = goodness.index(bestGoodness)

            # save best analysis result
            longestTime = generateIndividualTime(0.05, throttle, brake)
            time = generateTime(0.05, len(throttle))

            # plot best analysis result
            plotAnalysis(throttle, brake, throttleBrakeFun, longestTime, time, outputFolder)
            print(goodness[-1])
            print(filenames[i])
    
    print(goodness)

    script_end = datetime.now()
    print(script_end - script_start)