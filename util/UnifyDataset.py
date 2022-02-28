## UnifyDataset.py - Script that used in unification of data from YOLOv3 and CARLA simulator
## in one .csv file used for training of CNN neural network.

from pathlib import Path

errCode = -1 # error code

imgNameId = 0   # imgName index in both .csv files
throttleId = 1  # throttle index in cnn .csv file
breakId = 2     # break index in cnn .csv file 

noOfBoundingBoxesId = 1 # noOfBoundingBoxes index in yolov3 .csv file

yolov3OutCSVPath = '../camera_sensors_output/center/out_yolov3.csv' # paths to input .csv files
cnnOutCSVPath = '../camera_sensors_output/center/out_cnn.csv'       # paths to input .csv files
outCSVPath = '../camera_sensors_output/center/out.csv'              # path to output .csv files

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

def unifyCSVFiles(parsedYOLOv3csv, parsedCNNcsv):
    
    if(len(parsedYOLOv3csv) != len(parsedCNNcsv)):
        print("Dataset sizes are not equal! Cannot proceed with generation of dataset.")
        return errCode
    else:

        data = []

        # add header to data 
        headerStr = ['imgName', 'Vpresent', 'Vpl', 'Vpr', 'Vpc', 'safeToAcc']
        data.append(headerStr)

        correspondingImgFound = False

        for i in range(1, len(parsedYOLOv3csv)):
            # Done in slow manner because there can be a case where dataset from one 
            # .csv file doesn't follow names from the other .csv file in the same order
            # but the corresponding image can still be found in both.
            for j in range(1, len(parsedCNNcsv)):
            
                if(parsedYOLOv3csv[i][imgNameId] == parsedCNNcsv[j][imgNameId]):
                    correspondingImgFound = True
                    tempData = []

                    safeToAccelerate = 0
                    vehiclePresentLeft = 0
                    vehiclePresentRight = 0
                    vehiclePresentCenter = 0

                    # determine if it is safe to accelerate
                    # this parameter needs to be adjusted by hand 
                    # determined solely based on the throttle and break parameters 
                    # from parsedCNNcsv (may be wrongly determined and not fully accurate)
                    if(float(parsedCNNcsv[j][throttleId]) > 0 and float(parsedCNNcsv[j][breakId]) == 0):
                        safeToAccelerate = 1
                    elif(float(parsedCNNcsv[j][throttleId]) == 0 and float(parsedCNNcsv[j][breakId]) > 0):
                        safeToAccelerate = 0
                    else:
                        # undefined behvaiour 
                        # not safe to accelerate
                        safeToAccelerate = 0

                    tempData = [parsedYOLOv3csv[i][imgNameId], 1 if parsedYOLOv3csv[i][noOfBoundingBoxesId] else 0, vehiclePresentLeft, vehiclePresentRight, vehiclePresentCenter, safeToAccelerate]
                    
                    break

            if not correspondingImgFound:
                print("Corresponding image in two datasets not found! Datasets might not be constructed on the same set of images!")
                return errCode

            data.append(tempData)

    return data

def saveCSVFile(outPath, data):
    # Format: imgName, Vpresent, Vpl, Vpr, safeToAcc

    csvFile = open(outPath, "w+") 

    print("Saving .csv file to location", outPath)

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

    # unify .csv data
    unifiedData = unifyCSVFiles(yolov3Data, cnnData)

    # save .csv file
    saveCSVFile(outCSVPath, unifiedData)

if __name__ == "__main__":
    main()