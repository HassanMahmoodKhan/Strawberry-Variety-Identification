import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import onnxruntime as ort
from PIL import Image

from llm import *

# Get the current working directory
current_directory = os.getcwd()

# Construct the path to the 'dataset' directory
dataset_directory = os.path.join(current_directory, 'dataset')
# Check if the 'dataset' directory exists; if not, create it
os.makedirs(dataset_directory, exist_ok=True)

# Construct the path to the 'models' directory
models_directory = os.path.join(current_directory, 'models')
os.makedirs(models_directory, exist_ok=True)

class_names = ['1975', '269', 'Benadice', 'Fortuna', 'Monterey', 'Radiance', 'SanAndreas']

# Path to keras and onnx files
keras = os.path.join(models_directory, 'pretrained.keras')
onnx = os.path.join(models_directory, 'pretrained.onnx')

@st.cache_resource
def load_model(model_path):
    """
    This function loads the model from file and returns the model object.

    Parameters:
        model_path: The path where the model file is stored.

    Returns:
        model: The model object.
        model_type: .keras or .onnx
    """
    try: 
        # Check if path has .keras or .onnx file extension
        model_type = os.path.splitext(model_path)[1]

        if model_type == '.keras':

            print(f"Attempting to load model from '{model_path}'")
            # Load model from file
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
            return model, model_type

        elif model_type == '.onnx':
            print(f"Attempting to load model from '{model_path}'")
            # Create an ONNX runtime inference session
            model = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            # Fetch input and output tensor names
            print("Model loaded successfully.")
            return model, model_type
        else:
            raise ValueError(f"Unsupported model file extension: {model_type}")
        
    except FileNotFoundError:
        print(f"Error: File '{model_path}' not found.")
    except TypeError:
        print(f"Error: '{model_path}' is not a valid path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

@st.cache_data
def predict(model, model_type, image, variety, input_shape=(512,512)):
    """
    This function takes in an image and feeds it to the model for inferencing.

    Parameters:
        model: Pre-trained model object.
        model_type: The model type e.g., Keras or ONNX
        image: Input image.
        variety: The associated strawberry variety.
        input_shape: Tuple specifying the input shape expected by the model (height, width).

    Returns:
        predicted_label: The predicted strawberry variety by the model. 
    """
    # Check if input image and associated variety is provided
    if not image == None and not variety == None:
        # Resize and add batch dimension to the image
        image_resized = tf.image.resize(image, [input_shape[0], input_shape[1]])
        input_image = np.expand_dims(image_resized, axis=0)

    if model_type == '.keras':
        # Make predictions
        start_time = time.perf_counter()
        predictions = model.predict(input_image)
        st.write(f"Time to prediction: {(time.perf_counter() - start_time):.2f} seconds")

    else:
        # Fetch input and output tensor names
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        # Make predictions
        start_time = time.perf_counter()
        predictions = model.run([output_name], {input_name:input_image})
        st.write(f"Time to prediction: {(time.perf_counter() - start_time):.2f} seconds")

   # Return the predicted label     
    predicted_label = (np.argmax([predictions[0]]))
    return class_names[predicted_label]


#Add a title to the sidebar
add_title = st.markdown('# Strawberry Variety Identification')

# Add a selectbox for the runtime option
df =  pd.DataFrame(
    {'runtime': ['TensorFlow', 'ONNXRuntime']})

# Add a selectbox to the sidebar:
runtime_selection = st.sidebar.selectbox(
    'Choose Runtime',
    df['runtime']
)

st.sidebar.write(' ### Runtime: ', runtime_selection)

# Check what runtime option was selected by the user
if runtime_selection == 'TensorFlow':
    runtime = keras
else:
    runtime = onnx
    
# Upload image file
with st.expander("### Image Naming Instructions"):
    st.write("*variety.jpg/jpeg/png*")

add_image = st.file_uploader(label='Upload image here', type=['.jpg', '.jpeg', '.png'], label_visibility='visible')
if add_image is not None:
    image = Image.open(add_image)
    true_label = os.path.splitext(add_image)[0]
    st.image(image, caption='Uploaded Image.', width=300)
    st.success("Image uploaded successfully!")

# Upload image button
add_button = st.sidebar.button("Run", type='secondary', use_container_width=True)

if add_button is not None:
    
    # Load model from file
    model, model_type = load_model(runtime)
    # Perform Inference
    label_predicted = predict(model, model_type, image, true_label)

    st.write(f"True Label: {true_label}, Predicted Label: {label_predicted}")


# # Invoke OpenAI's LLM for query answering related to the strawberry variety predicted
# invokeLLM(label_predicted)