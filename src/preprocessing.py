import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from path import Path

from main import dataset_directory, class_names

def process_image(image_path=None, variety=None, input_shape=(512,512)):
    """
    Loads an image from either the specified path or randomly from the provided dataset.

    Parameters:
        image_path: The path to the image.
        variety: The associated strawberry variety.
        input_shape: The input image shape

    Returns:
        image: The image.
        variety: The associated strawberry variety.
    """

    if not image_path and not variety:
        # Choose a random image from the dataset
        seed = np.random.randint(0,7)
        variety = class_names[seed]
        images = list(Path(dataset_directory).glob(f"{variety}/*"))
        image = cv2.imread(str(images[0]))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = tf.image.resize(image_rgb, [input_shape[0], input_shape[1]])  # Resize image
        input_image = np.expand_dims(image_resized, axis=0) # Model expects batch dimension 
    else:
        # Load image from the path provided
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = tf.image.resize(image_rgb, [input_shape[0], input_shape[1]])  # Resize image
        input_image = np.expand_dims(image_resized, axis=0) # Model expects batch dimension
    
    return input_image, variety

def display_image(image, variety):
    """
    Displays a single image and its associated variety.

    Parameters: 
        image: The image to display.
        variety: The corresponding strawberry variety.
    
    Returns:
        None
    """
    plt.figure(figsize=(6,6))
    image = image / 255.0
    plt.imshow(image)
    plt.title(f"{variety} Strawberry")
    plt.show()

def plot_curves(history, fig_path):
    """
    Plots the training and validation accuracy and loss curves.

    Parameters:
        history: The object returned after performing model.fit().
        fig_path: Path to save the figure at.

    Returns:
        None
    """
    print("Plotting accuracy and loss curves")

    # Plot and save the training and validation accuracy curves
    plt.figure(figsize=(10,8))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Save the plot
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + '/accuracy_loss.jpg')
    
    # Display the plot
    plt.show()