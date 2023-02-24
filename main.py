import glob
import pathlib
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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def resize_image(image, size=(200, 200)):
    return cv2.resize(image, size)


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def normalize_image(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def save_image(image, directory, filename):
    cv2.imwrite(os.path.join(directory, filename), image)

def get_random_image(root_directory):
    # Get a random subdirectory
    subdirectories = [os.path.join(root_directory, d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
    subdirectory = random.choice(subdirectories)

    # Get a random image from the subdirectory
    images = [os.path.join(subdirectory, f) for f in os.listdir(subdirectory) if os.path.isfile(os.path.join(subdirectory, f))]
    image_path = random.choice(images)

    return image_path
def process_images():
    # Get the directory to process
    directory = input("Enter the directory to process: ")
    people_count = len(os.listdir(directory))
    print("Found images of", people_count, "people/classes/labels in this directory")

    # Get the directory to save the images
    save_directory = input("Enter the directory to save the images: ")

    # Size to resize the images
    img_width, img_height = 200, 200

    # Initialize variables to keep track of the current person and folder
    current_person = None
    current_folder = None

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
                # Get the person's name from the filepath
                person = os.path.basename(root)

                # Check if we've encountered a new person
                if person != current_person:
                    # If so, create a new folder for the person's images
                    current_person = person
                    current_folder = os.path.join(save_directory, person)
                    os.makedirs(current_folder, exist_ok=True)

                # Load the image
                image = mpimg.imread(filepath)

                # Resize the image
                image = resize_image(image, (img_width, img_height))

                # Convert the image to grayscale
                image = convert_to_grayscale(image)

                # Normalize the image
                image = normalize_image(image)

                # Save the image in the current person's folder
                print("Saving image named", filename, "to", current_folder)
                #save_image(image, current_folder, filename)


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
directory_detection = input("From which directory would you like to detect a face?")
print("Detecting a random images' face...")
pixels = extract_face(get_random_image(directory_detection))
plt.imshow(pixels)
plt.show()
print(pixels.shape)

"""def load_face(dir):
    faces = list()
    # enumerate files
    for filename in os.listdir(dir):
        path = dir + filename
        face = extract_face(path)
        faces.append(face)
    return faces

def load_dataset(dir):
    # list for faces and labels
    X, y = list(), list()
    for subdir in os.listdir(dir):
        path = dir + subdir + '/'
        faces = load_face(path)
        labels = [subdir for i in range(len(faces))]
        print("loaded %d sample for class: %s" % (len(faces),subdir) ) # print progress
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)

trainX, trainy = load_dataset('D:\\pyCharm Projects Python\\database\\new_train\\')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('D:\\pyCharm Projects Python\\database\\new_train\\')
print(testX.shape, testy.shape)

np.savez_compressed('classes_pins_dataset_train.npz', trainX, trainy, testX, testy)"""



