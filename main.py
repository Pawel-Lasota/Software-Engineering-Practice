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
import tensorflow as tf
from numpy import asarray
import face_recognition
import matplotlib.image as mpimg
from sklearn.svm import SVC
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
    subdirectories = [os.path.join(root_directory, d) for d in os.listdir(root_directory) if
                      os.path.isdir(os.path.join(root_directory, d))]
    subdirectory = random.choice(subdirectories)

    # Get a random image from the subdirectory
    images = [os.path.join(subdirectory, f) for f in os.listdir(subdirectory) if
              os.path.isfile(os.path.join(subdirectory, f))]
    image_path = random.choice(images)

    return image_path


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

    if not results:
        return None

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


def process_images():
    # Get the directory to process
    directory = input("Enter the directory to process: ")
    people_count = len(os.listdir(directory))
    print("Found images of", people_count, "people/classes/labels in this directory")

    # Get the directory to save the images
    save_directory = input("Enter the directory to save the images (choose different than previous): ")

    # Create the overall train and test directories
    train_dir = os.path.join(save_directory, "train")
    test_dir = os.path.join(save_directory, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Size to resize the images
    img_width, img_height = 200, 200

    # Iterate through the directory
    for root, _, files in os.walk(directory):
        # Ignore the root directory
        if root == directory:
            continue

        # Get the person's name from the folder path
        person = os.path.basename(root)

        # Create the train and test directories for this person
        person_train_dir = os.path.join(train_dir, person)
        person_test_dir = os.path.join(test_dir, person)
        os.makedirs(person_train_dir, exist_ok=True)
        os.makedirs(person_test_dir, exist_ok=True)
        # Count the number of images for this person
        image_count = len(files)
        print("Processing", person, "with", image_count, "images")

        # Split the images into train and test sets
        train_count = int(image_count * 0.8)
        test_count = image_count - train_count

        # Iterate through the images
        for i, file in enumerate(files):
            # Get the file path
            file_path = os.path.join(root, file)

            # Extract the face from the image
            face = extract_face(file_path, required_size=(img_width, img_height))

            # If no face is detected, skip the image
            if face is None:
                print("No face detected in", file_path)
                continue

            # Preprocess the image
            face = resize_image(face, size=(img_width, img_height))
            face = convert_to_grayscale(face)
            face = normalize_image(face)

            # Decide whether to save to the train or test directory
            if i < train_count:
                save_image(face, person_train_dir, file)
            else:
                save_image(face, person_test_dir, file)

        print("Finished processing", person)

    print("Processing complete!")

process_images()
