from PIL import Image
import Logger
import numpy as np

# Properties
BYTES_PER_IMAGE = 784

# If loading from local file.
def getData(file):
    # Read bytes from file.
    f = open(file, 'rb')
    return f.read()

# Flip byte from 255 -> 0, 0 -> 255, e.t.c
def flipByte(imageData):
    # Convert to byte array to allow data manipulation
    byteArray = bytearray(imageData)
    b = 0
    # Foreach byte in the image
    while (b < len(byteArray)):
        # Flip byte colours.
        byteArray[b] = 255 - byteArray[b]
        b += 1
        
    # Convert back to bytes.
    return bytes(byteArray)

def seperateImages(data):
    output = []
    start = 80
    for i in range(start, len(data), BYTES_PER_IMAGE):
        # i is looping through all the bytes, but I want the index of the specific image I'm on, so divide by amount of bytes in the images
        outputIndex = round(i / BYTES_PER_IMAGE)
        # Append the bytes of the current image
        output.append(data[i:(i + BYTES_PER_IMAGE)])
        # Flip the bytes of the current image.
        output[outputIndex] = flipByte(output[outputIndex])
        Logger.Log("Finished Processing Image #" + str(outputIndex + 1) + " (" + str(i + BYTES_PER_IMAGE) + "bytes processed)")
    return output

def saveImage(name, data):
    im = Image.frombytes('L', (28, 28), data)
    im.save(name)

def saveImagesToDirectory(directory):
    i = 0
    while (i < len(output)):
        saveImage("image_test/temp" + str(i) + ".png", output[i])
        i += 1
