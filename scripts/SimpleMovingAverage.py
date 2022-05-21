''' 
    SimpleMovingAverage.py - is a script that handles SMA buffers, calculates SMA based on those buffers and provides
                             interfaces for storing and reading values of SMA data
'''

import numpy as np

class SimpleMovingAverage:

    def __init__(self, bufferSize, dataSize):
        self.bufferSize = bufferSize
        self.dataSize = dataSize
        self.dataBuffer = np.zeros(shape=(self.bufferSize, self.dataSize))
        self.smaBuffer = np.zeros(shape=(self.dataSize))

    def addToBuffer(self, data):

        self.dataBuffer = np.insert(self.dataBuffer, 0, np.array(data), axis=0)
        self.dataBuffer = self.dataBuffer[:self.bufferSize]

        self.__calculate_sma__()

    def __calculate_sma__(self):
        
        lSMABuffer = np.transpose(self.dataBuffer)

        for i in range(self.dataSize):
            sma = np.sum(lSMABuffer[i]) / self.bufferSize
            self.smaBuffer[i] = sma

    def getSMABuffer(self):
        return self.smaBuffer

    def getDataBuffer(self):
        return self.dataBuffer