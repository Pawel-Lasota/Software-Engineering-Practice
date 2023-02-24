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

    # Initialize lists to hold the training and testing data
    training_data = []
    testing_data = []

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

                # Randomly split the data into training and testing sets
                if random.random() < 0.8:
                    # Save the image in the current person's training folder
                    training_data.append((image, current_folder))
                else:
                    # Save the image in the current person's testing folder
                    testing_data.append((image, current_folder))

    # Shuffle the training and testing data
    random.shuffle(training_data)
    random.shuffle(testing_data)

    # Save the training data
    for i, (image, folder) in enumerate(training_data):
        filename = f"train_{i}.jpg"
        save_image(image, folder, filename)

    # Save the testing data
    for i, (image, folder) in enumerate(testing_data):
        filename = f"test_{i}.jpg"
        save_image(image, folder, filename)


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

