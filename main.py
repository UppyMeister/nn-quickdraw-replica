#import numpyToBinary as npybin
import ImageManipulator
import ImageData
import Logger
from NeuralNetwork import NeuralNetwork
import urllib.request
import random

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

def startNetwork():
    nn = NeuralNetwork(2, 2, 2)
    inputs = [1, 0]
    targets = [1, 0]
    #nn.feedforward(inputs)
    nn.train(inputs, targets)

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

    print(nn.feedforward([1, 0]))
    print(nn.feedforward([0, 1]))
    print(nn.feedforward([0, 0]))
    print(nn.feedforward([1, 1]))

def prepareImageData(category, rawData, label, limit):
    # No longer required, as the numpy file can be read directly from the site
    # into the seperateImages, as long as the 80 header bytes are accounted for.
    #npybin.convertImagesToBinary("data/npy/" + imageCategory + ".npy", count)
    contentLength = int(rawData.headers['content-length'])
    if (limit == None or NUMPY_HEADER_BYTES + (limit * BYTES_PER_IMAGE) > contentLength):
        limit = int((contentLength - NUMPY_HEADER_BYTES) / BYTES_PER_IMAGE)
        Logger.Log("Image count set to max.", "INFO")
    Logger.Log("Loading data...\n" + str(rawData.headers), "INFO")
    imageData = ImageManipulator.seperateImages(rawData.read(NUMPY_HEADER_BYTES + (BYTES_PER_IMAGE * limit)))
    Logger.Log("Finished loading data.", "INFO")
    completeImageData = []
    for x in range(0, len(imageData)):
        completeImageData.append(ImageData.ImageData(imageData[x], label))
    threshold = round(0.8 * len(completeImageData))
    category.training = completeImageData[0:threshold]
    category.testing = completeImageData[threshold:len(imageData)]
    ImageManipulator.saveImage("first.png", imageData[0])
    ImageManipulator.saveImage("last.png", imageData[len(imageData)-1])
    ImageManipulator.saveImage("random.png", imageData[random.randint(0, len(imageData)-1)])

#prepareImageData(data_rainbow, rawData_rainbow, rainbow, 1000)
#prepareImageData(data_anvil, rawData_anvil, anvil, 1000)
#prepareImageData(data_ambulance, rawData_ambulance, ambulance, 1000)

testNetwork()
