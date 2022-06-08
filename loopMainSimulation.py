from MainSimulation import main, calculateGridSearchIndexes

gPIDParameters = [[100000, 10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001], # Kp
                  [100000, 10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001], # Ki
                  [100000, 10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]] # Kd

if __name__ == '__main__':

    # loop trough MainSimulation 1000 times for PID Longitudinal Controller Kp, Ki, Kd grid search

    ii = 5
    jj = 5 
    kk = 6

    for i in range(1):
        main(ii, jj, kk)
        #temp = calculateGridSearchIndexes(gPIDParameters, ii, jj, kk)
        #ii, jj, kk = temp[0], temp[1], temp[2]