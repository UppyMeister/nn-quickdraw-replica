import json
import random
import os.path
import urllib.request
import ImageHandler
from Logger import Logger
from LogLevel import LogLevel
from ImageData import ImageData
from NeuralEncoder import NeuralEncoder
from NeuralNetwork import NeuralNetwork

categories = [{"category": "aircraft carrier", "limit": None},
              {"category": "airplane", "limit": None},
              {"category": "alarm clock", "limit": None},
              {"category": "ambulance", "limit": None},
              {"category": "angel", "limit": None},
              {"category": "ant", "limit": None},
              {"category": "anvil", "limit": None},
              {"category": "apple", "limit": None},
              {"category": "axe", "limit": None},
              {"category": "banana", "limit": None},
              {"category": "baseball bat", "limit": None},
              {"category": "bicycle", "limit": None},
              {"category": "cat", "limit": None}]

data_objects = [lambda: None for x in categories]

# Properties
BYTES_PER_IMAGE = 784
NUMPY_HEADER_BYTES = 80

Logger = Logger(LogLevel.INFO)

def calculateHiddenLayerNodeCount():
    # formula = nsamples / (alpha * (ninputs + noutputs))
    samples = 0
    for x in data_objects:
        samples += len(x.testing)
    return int(samples / (3 * (BYTES_PER_IMAGE + len(categories))))

def getTrainingData():
    training = []
    for x in data_objects:
        training.extend(x.training)
    random.shuffle(training)
    return training

def getTestingData():
    testing = []
    for x in data_objects:
        testing.extend(x.testing)
    random.shuffle(testing)
    return testing

def trainOneEpoch(network, training):
    Logger.Log("Beginning training with " + str(len(training)) + " items.", LogLevel.INFO)
    for i in training:
        data = i.data
        inputs = [x / 255 for x in data]
        targets = [0] * len(categories)
        targets[i.label] = 1
        network.train(inputs, targets)

def testAll(network, testing):
    Logger.Log("Beginning testing with " + str(len(testing)) + " items.", LogLevel.INFO)
    correct = 0
    for i in range(len(testing)):
        data = testing[i].data
        inputs = [x / 255 for x in data]
        results = network.predict(inputs)
        guess = results.index(max(results))
        #Logger.Log("RESULT: " + str(results) + "\nGUESS: " + str(guess) + ", ACTUAL: " + str(label))
        if (guess == testing[i].label):
            correct += 1
    #Logger.Log("Testing Complete.", LogLevel.INFO)
    percent_correct = (correct / len(testing)) * 100
    Logger.Log("Success Rate: " + str(round(percent_correct, 2)) + "%", LogLevel.INFO)

def mainNetwork(network, info):
    training = getTrainingData()
    testing = getTestingData()
    testAll(network, testing) # Initial Test
    for i in range(0, info["epochs"]):
        trainOneEpoch(network, training)
        Logger.Log("Trained for " + str(i + 1) + " epoch" + ("s" if i > 0 else ""), LogLevel.INFO)
        Logger.Log("Saving Network", LogLevel.INFO)
        encoded = json.dumps([{"info": info}, {"network": network.__dict__}], separators=(',',': '), sort_keys=True, indent=4, cls=NeuralEncoder)
        with open(info["name"] + ".json", "w") as f:
            f.write(encoded)
            f.close()
        
        testAll(network, testing)

def startNetwork(name, epochs):
    nn = NeuralNetwork(BYTES_PER_IMAGE, calculateHiddenLayerNodeCount(), len(categories))
    info = {"name": name, "epochs": epochs}
    mainNetwork(nn, info)

def loadNetwork(json):
    nn = NeuralNetwork(json["network"])
    mainNetwork(nn, json["info"])

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

def prepareImageData(category, rawData, label, customTestingFiles, limit):
    # No longer required, as the numpy file can be read directly from the site
    # into the seperateImages, as long as the 80 header bytes are accounted for.
    #npybin.convertImagesToBinary("data/npy/" + imageCategory + ".npy", count)
    Logger.Log("Loading data for \"" + category.category + "\"", LogLevel.INFO)
    
    contentLength = int(rawData.headers['content-length'])
    if (limit == None or NUMPY_HEADER_BYTES + (limit * BYTES_PER_IMAGE) > contentLength):
        limit = int((contentLength - NUMPY_HEADER_BYTES) / BYTES_PER_IMAGE)
        Logger.Log("Image count set to max (" + str(int((contentLength - NUMPY_HEADER_BYTES) / BYTES_PER_IMAGE)) + " images).", LogLevel.INFO)

    try:
        imageData = ImageHandler.seperateImages(rawData.read(NUMPY_HEADER_BYTES + (BYTES_PER_IMAGE * limit)))
    except Exception as e:
        Logger.Log("Error loading data: " + str(e))
    
    Logger.Log("Finished loading data.", LogLevel.INFO)
    
    completeImageData = [ImageData(x, label) for x in imageData]
    threshold = round(0.8 * len(completeImageData))
    category.training = completeImageData[0:threshold]
    category.testing = completeImageData[threshold:len(imageData)]
    category.custom_testing = customTestingFiles

def loadData():
    for i in range(len(categories)):
        url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/" + categories[i]["category"].replace(" ", "%20") + ".npy"
        try:
            d = urllib.request.urlopen(url)
        except Exception as e:
                Logger.Log("Failed to load info from url (" + url + "): " + str(e))
                print(categories[i]["category"] + " does not exist")
                continue

        data_objects[i].category = categories[i]["category"]
        prepareImageData(data_objects[i],
                         d,
                         i,
                         ImageHandler.createImagesFor(ImageHandler.getFilesFromDirectory("custom_testing_data/" + categories[i]["category"]), i),
                         categories[i]["limit"])

def main():
    loadData()
    choice = input("Neural Network\n[1] New Network\n[2] Load Network\n=> ")
    if (choice == "1"):
        name = input("Network Name\n=> ")
        epochs = input("Epochs\n=> ")
        if (epochs.isdigit()):
            startNetwork(name, int(epochs))
        else:
            print("Epochs must be a number")
            main()
    elif (choice == "2"):
        name = input("Network to load\n=> ")
        if (os.path.isfile(name + ".json")):
            with open("main.json", "r") as f:
                data = f.read()
                loadNetwork(json.loads(data))
        else:
            print("No network called: " + name)
            main()
    else:
        print("Invalid choice")
        main()
main()
