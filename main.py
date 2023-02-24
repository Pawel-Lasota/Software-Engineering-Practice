import glob
import random
import shutil
from tkinter import Image
import os

import cv2
import mtcnn
from matplotlib import pyplot, pyplot as plt
from mtcnn import MTCNN
from PIL import Image
import numpy as np
from numpy import asarray
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
# example of creating a face embedding

# load image from file
RootDir = "D:\pyCharm Projects Python\database\classes_pins_dataset_train"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def resize_image(image, size=(200, 200)):
    return cv2.resize(image, size)


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def normalize_image(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def save_image(image, directory, filename):
    cv2.imwrite(os.path.join(directory, filename), image)

def process_images():
    # Get the directory to process
    directory = input("Enter the directory to process: ")
    people_count = len(os.listdir(directory))
    print("Found images of", people_count, "people/classes/labels in this directory")

    # Get the directory to save the images
    save_directory = input("Enter the directory to save the images: ")

    # Size to resize the images
    img_width, img_height = 200, 200

    # Iterate through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Get the filepath
            filepath = os.path.join(root, file)

            # Get the filename
            filename = os.path.basename(filepath)

            # Get the extension
            extension = os.path.splitext(filepath)[1]


            # Check if the file is an image
            if extension in [".jpg", ".jpeg", ".png"]:
                # Load the image
                image = mpimg.imread(filepath)

                # Resize the image
                image = resize_image(image, (img_width, img_height))

                # Convert the image to grayscale
                image = convert_to_grayscale(image)

                # Normalize the image
                image = normalize_image(image)

                # Save the image
                print("Saving image named", filename,"to new folder...")
                #save_image(image, save_directory, filename)


if __name__ == "__main__":
    process_images()

# extract a single face from a given photograph
def extract_face(filename, required_size=(200, 200)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # deal with negative pixel index
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

# load the photo and extract the face
#directory_detection = input("From which directory would you like to detect a face?")
pixels = extract_face("D:\\pyCharm Projects Python\\database\\new_dataset\\Hugh Jackman0_1271.jpg")
plt.imshow(pixels)
plt.show()
print(pixels.shape)


""""
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
print("Split the people into", n_train, "for training and", n_test, "for testing.")"""


