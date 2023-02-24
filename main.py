from tkinter import Image

import mtcnn
from matplotlib import pyplot
from mtcnn import MTCNN
from PIL import Image
import numpy as np
from numpy import asarray
# example of creating a face embedding

# load image from file
pixels = pyplot.imread("D:\pyCharm Projects Python\database\classes_pins_dataset_train\pins_jeff bezos\jeff bezos0_2040.jpg")
RootDir = "D:\pyCharm Projects Python\database\classes_pins_dataset_train"
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
results = detector.detect_faces(pixels)

# extract the bounding box from the first face
x1, y1, width, height = results[0]['box']
x2, y2 = x1 + width, y1 + height

# extract the face
face = pixels[y1:y2, x1:x2]


# resize pixels to the model size
image = Image.fromarray(face)
image = image.resize((224, 224))
face_array = asarray(image)


# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# load the photo and extract the face
pixels = extract_face("D:\pyCharm Projects Python\database\classes_pins_dataset_train\pins_jeff bezos\jeff bezos0_2040.jpg")
# plot the extracted face
pyplot.imshow(pixels)
# show the plot
pyplot.show()

def index_and_split_data(percentage):
    # Get data
    people_count = len(os.listdir(RootDir))
    print("Found images of", people_count, "people or classes/labels.")
    train_index, test_index = {}, {}
    n_train, n_test = 0, 0
    total = 0
    for name in os.listdir(RootDir):
        path = RootDir + "/" + name
        n = len(os.listdir(path))
        total += n
        if np.random.rand() < percentage:
            train_index[name] = n
            n_train += 1
        else:
            test_index[name] = n
            n_test += 1
    print("Found a total of", total, "images.")
    return n_train, train_index, n_test, test_index


# Run the previous function
n_train, train_list, n_test, test_list = index_and_split_data(0.8)
print("Split the people into", n_train, "for training and", n_test, "for testing.")


