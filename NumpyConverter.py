import numpy as np

def loadFile(file):
    return np.load(file)

def writeToFile(filename, data):
    f = open("data/bin/" + filename + ".bin", "wb")
    i = 0
    while (i < len(data)):
        f.write(bytearray(data[i]))
        i += 1

def convertImagesToBinary(file, count):
    data = loadFile(file)
    output = []
    n = 0
    while (n < count):
        output.append(data[n].tolist())
        n += 1
    writeToFile("aircraft" + str(count), output)
    print("Converted npy to bin.\noutput -> " + "aircraft" + str(count) + ".bin")
