import os
import logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import time
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import onnxruntime as ort

from preprocessing import process_image

def evaluate(model, model_type, test_dataset, class_names, assets_directory, input_shape=(512,512)):
    """
    This function evaluates the model on the test dataset and reports metrics such as loss, accuracy, inference time, etc.

    Parameters:
        model: The keras model object.
        model_type: The type of model i.e., custom, pretrained, etc.
        test_dataset: The 'test' TensorFlow data object. 
        class_names: The labels for the dataset.
        assets_directory: The path to the assets directory. 
        input_shape: The input image shape.
    
    Returns:
        None
    """
    start_time = time.time_ns()
    test_loss, test_accuracy = model.evaluate(test_dataset)
    end_time = time.time_ns()

    with open(os.path.join(assets_directory, f'{model_type}/log.txt'), 'w') as file:
        file.write(f"Time to Evaluate: {end_time-start_time:.4f}\n")
        file.write(f"Test Loss: {test_loss:.4f}\n")
        file.write(f"Test Accuracy: {test_accuracy:.4f}\n")

    true_labels = []
    predicted_labels = []
    for images, labels in test_dataset:
        # 'images' is a single image batch
        # 'labels' is a single label batch
        for image, label in zip(images, labels):
            # 'image' is single image tensor
            # 'label' is single label (categorical)
            
            # Convert categorical labels to class indices
            true_labels.append(np.argmax(label.numpy()))
            
            # Convert image tensor to a NumPy array
            image_np = image.numpy()
            image_resized = tf.image.resize(image_np, [input_shape[0], input_shape[1]])  # Resize image
            input_image = np.expand_dims(image_resized, axis=0) # Model expects batch dimension

            prediction = model.predict(input_image)

            # Obtaining the index of the highest probability class
            predicted_label = np.argmax(prediction, axis=1)[0]
            predicted_labels.append(predicted_label)

    # Ensure the labels are flattened lists
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Compute confusion matrix
    cm = tf.math.confusion_matrix(true_labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    
    # Save the plot
    fig_path = os.path.join(assets_directory, f'{model_type}')
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + '/confusion_matrix.jpg')
    
    # Display the plot
    plt.show()

def predict(model_path, dataset_directory, class_names, image_path=None, variety=None, input_shape=(512,512)):
    """
    This function takes in an image and feeds it to the model for prediction purposes.

    Parameters:
        model_path: Path to the pre-trained Keras/ONNX model.
        dataset_directory: Path to the dataset directory.
        class_names: The dataset labels.
        image_path: Path to the input image.
        variety: The associated strawberry variety.
        input_shape: Tuple specifying the input shape expected by the model (height, width).

    Returns:
        predicted_label: The predicted strawberry variety by the model. 
    """
    # Check if input image and associated variety is provided
    if not image_path == None and not variety == None:
        # Read and preprocess the image
        input_image, label = process_image(dataset_directory, class_names, image_path, variety, input_shape)
    else: 
        # Choose a random image from the dataset for prediction
        input_image, label = process_image(dataset_directory, class_names)

    try:
        # Check if path has .keras or .onnx file extension
        file_extension = os.path.splitext(model_path)[1]

        if file_extension == '.keras':

            print(f"Attempting to load model from '{model_path}'")
            # Load model from file
            loaded_model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
            # Make predictions
            start_time = time.perf_counter()
            predictions = loaded_model.predict(input_image)
            print(f"Time to prediction: {(time.perf_counter() - start_time):.2f} seconds")

        elif file_extension == '.onnx':

            print(f"Attempting to load model from '{model_path}'")
            # Create an ONNX runtime inference session
            sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            # Fetch input and output tensor names
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            print("Model loaded successfully.")
            # Make predictions
            start_time = time.perf_counter()
            predictions = sess.run([output_name], {input_name:input_image})
            print(f"Time to prediction: {(time.perf_counter() - start_time):.2f} seconds")

        else:
            raise ValueError(f"Unsupported model file extension: {file_extension}")
        
        # Plot the input image with the predicted label
        plt.figure(figsize=(8,6))
        input_image = tf.cast(input_image[0], tf.float32) / 255.0  # Rescale for plotting
        plt.imshow(input_image)
        predicted_label = (np.argmax([predictions[0]]))
        plt.title(f'True Label: {label}, Predicted Label: {class_names[predicted_label]}')
        plt.axis('off')
        plt.show()

        return class_names[predicted_label]
    
    except FileNotFoundError:
        print(f"Error: File '{model_path}' not found.")
    except TypeError:
        print(f"Error: '{model_path}' is not a valid path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':

    model_path = r"D:\Umer Project - Strawberry Identification App\models\custom.onnx"
    image_path = r"D:\Umer Project - Strawberry Identification App\test_images\Monterey_0502_13.jpg"
    variety = 'Moneterey'
    predict(model_path, image_path, variety)