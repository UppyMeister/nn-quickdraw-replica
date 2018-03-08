import random
import urllib.request

import Logger
import ImageData
import ImageDataHandler
from NeuralNetwork import NeuralNetwork

# Properties
BYTES_PER_IMAGE = 784
NUMPY_HEADER_BYTES = 80

# Objects
data_rainbow = lambda: None
data_anvil = lambda: None
data_ambulance = lambda: None

# Raw Data
rawData_rainbow = urllib.request.urlopen("https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/rainbow.npy")
rawData_anvil = urllib.request.urlopen("https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/anvil.npy")
rawData_ambulance = urllib.request.urlopen("https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/ambulance.npy")

# Labels
rainbow = 0
anvil = 1
ambulance = 2

def getTrainingData():
    training = []
    training.extend(data_rainbow.training)
    training.extend(data_anvil.training)
    training.extend(data_ambulance.training)
    random.shuffle(training)
    return training

def getTestingData():
    testing = []
    testing.extend(data_rainbow.testing)
    testing.extend(data_anvil.testing)
    testing.extend(data_ambulance.testing)
    random.shuffle(testing)
    return testing

def trainOneEpoch(network, training):
    Logger.Log("Beginning training with " + str(len(training)) + " items.", "INFO")
    for i in range(0, len(training)):
        data = training[i].data
        inputs = [x / 255 for x in data]
        label = training[i].label
        targets = [0, 0, 0]
        targets[label] = 1
        network.train(inputs, targets)

def testAll(network, testing):
    Logger.Log("Beginning testing with " + str(len(testing)) + " items.", "INFO")
    correct = 0
    for i in range(0, len(testing)):
        data = testing[i].data
        label = testing[i].label
        inputs = [x / 255 for x in data]
        results = network.predict(inputs)
        guess = results.index(max(results))
        #Logger.Log("RESULT: " + str(results) + "\nGUESS: " + str(guess) + ", ACTUAL: " + str(label))
        if (guess == label):
            correct += 1
    #Logger.Log("Testing Complete.", "INFO")
    percent_correct = (correct / len(testing)) * 100
    Logger.Log("Success Rate: " + str(percent_correct) + "%", "INFO")

def startNetwork(epochs):
    nn = NeuralNetwork(784, 64, 3)
    training = getTrainingData()
    testing = getTestingData()
    testAll(nn, testing) # Initial Test
    for i in range(0, epochs):
        trainOneEpoch(nn, training)
        Logger.Log("Trained for " + str(i + 1) + " epoch" + ("s" if i > 0 else ""), "INFO")
        testAll(nn, testing)

def testNetwork():
    # Test using the XOR problem
    training_data = [
        {
            "inputs": [0, 1],
            "targets": [1]
        },
        {
            "inputs": [1, 0],
            "targets": [1]
            },
            {
            "inputs": [0, 0],
            "targets": [0]
            },
            {
            "inputs": [1, 1],
            "targets": [0]
            }
        ]
    nn = NeuralNetwork(2, 2, 1)

    for i in range(0, 1000000):
        for x in training_data:
            nn.train(x["inputs"], x["targets"])

    print(nn.predict([1, 0]))
    print(nn.predict([0, 1]))
    print(nn.predict([0, 0]))
    print(nn.predict([1, 1]))

def prepareImageData(category, rawData, label, limit):
    # No longer required, as the numpy file can be read directly from the site
    # into the seperateImages, as long as the 80 header bytes are accounted for.
    #npybin.convertImagesToBinary("data/npy/" + imageCategory + ".npy", count)
    contentLength = int(rawData.headers['content-length'])
    if (limit == None or NUMPY_HEADER_BYTES + (limit * BYTES_PER_IMAGE) > contentLength):
        limit = int((contentLength - NUMPY_HEADER_BYTES) / BYTES_PER_IMAGE)
        Logger.Log("Image count set to max (" + str((contentLength - NUMPY_HEADER_BYTES) / BYTES_PER_IMAGE) + " images).", "INFO")
    #Logger.Log("Loading data...\n" + str(rawData.headers), "INFO")
    Logger.Log("Loading data...", "INFO")
    imageData = ImageDataHandler.seperateImages(rawData.read(NUMPY_HEADER_BYTES + (BYTES_PER_IMAGE * limit)))
    Logger.Log("Finished loading data.", "INFO")
    completeImageData = []
    for x in range(0, len(imageData)):
        completeImageData.append(ImageData.ImageData(imageData[x], label))
    threshold = round(0.8 * len(completeImageData))
    category.training = completeImageData[0:threshold]
    category.testing = completeImageData[threshold:len(imageData)]

prepareImageData(data_rainbow, rawData_rainbow, rainbow, 5000)
prepareImageData(data_anvil, rawData_anvil, anvil, 5000)
prepareImageData(data_ambulance, rawData_ambulance, ambulance, 5000)

startNetwork(300)
