#import numpyToBinary as npybin
import imageManipulator as im
import urllib.request
import random

# Properties
BYTES_PER_IMAGE = 784
NUMPY_HEADER_BYTES = 80

# Raw Data
rawData_rainbow = urllib.request.urlopen("https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/rainbow.npy")
rawData_anvil = urllib.request.urlopen("https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/anvil.npy")

# Objects
data_rainbow = lambda: None
data_anvil = lambda: None

def prepareImageData(category, rawData, limit):
    # No longer required, as the numpy file can be read directly from the site
    # into the seperateImages, as long as the 80 header bytes are accounted for.
    #npybin.convertImagesToBinary("data/npy/" + imageCategory + ".npy", count)
    imageData = im.seperateImages(rawData_rainbow.read(NUMPY_HEADER_BYTES + (BYTES_PER_IMAGE * limit)))
    threshold = round(0.8 * len(imageData))
    category.training = imageData[0:threshold]
    category.testing = imageData[threshold:len(imageData)]
    #im.saveImage("first.png", imageData[0])
    #im.saveImage("last.png", imageData[len(imageData)-1])
    #im.saveImage("random.png", imageData[random.randint(0, len(imageData)-1)])

prepareImageData(data_rainbow, rawData_rainbow, 1000)
