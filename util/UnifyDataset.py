## UnifyDataset.py - Script that used in unification of data from YOLOv3 and CARLA simulator
## in one .csv file used for training of CNN neural network.

from pathlib import Path

# paths to input .csv files
yolov3OutCSVPath = '../camera_sensors_output/test/out_yolov3.csv'
cnnOutCSVPath = '../camera_sensors_output/test/out_cnn.csv'
# paths to output .csv files
outCSVPath = '../camera_sensors_output/test/out.csv'

def parseCSVFile(csvPath):
    data = []
    
    csv = open(Path(csvPath), "r") 
    csvFile = csv.readlines()

    for line in csvFile:
        if(len(line) > 0):
            data.append(line.replace('\n', '').split(','))

    csv.close()

    print(data)
    return data

def unifyCSVFiles(parsedYOLOv3csv, parsedCNNcsv):
    pass

def saveCSVFile(outPath, data):
    # Format: imgName, noOfBoundingBoxes, throttle, break

    csvFile = open(outPath, "w+") 

    for i in range(len(data)):
        line = ''
        for j in range(len(data[i])):
            line += data[i][j] + ','

        line += '\n'

        csvFile.write(line)

    csvFile.close()

def main():
    
    # parse YOLOv3 .csv file
    yolov3Data = parseCSVFile(yolov3OutCSVPath)

    # parse CNN .csv file
    cnnData = parseCSVFile(cnnOutCSVPath)

    # unify .csv data
    unifiedData = unifyCSVFiles(yolov3Data, cnnData)

    # save .csv file
    mockData = [['imgName', 'noOfBoundingBoxes', 'throttle', 'break'], ['0', '0', '0', '0']]
    saveCSVFile(outCSVPath, mockData)

if __name__ == "__main__":
    main()