import os
import logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import time
import argparse

from preprocessing import *
from data_loader import load_datasets
from train import *
from test import *
from onnx_conv import *
from llm import *

# Get the current working directory
current_directory = os.getcwd()

# Construct the path to the 'dataset' directory
dataset_directory = os.path.join(current_directory, 'dataset')
# Check if the 'dataset' directory exists; if not, create it
os.makedirs(dataset_directory, exist_ok=True)

# Construct the path to the 'assets' directory
assets_directory = os.path.join(current_directory, 'assets')
# Check if the 'assets' directory exists; if not, create it
os.makedirs(assets_directory, exist_ok=True)

# Construct the path to the 'models' directory
models_directory = os.path.join(current_directory, 'models')
# Check if the 'assets' directory exists; if not, create it
os.makedirs(models_directory, exist_ok=True)

class_names = ['1975', '269', 'Benadice', 'Fortuna', 'Monterey', 'Radiance', 'SanAndreas']

def main():

    parser = argparse.ArgumentParser(description="A script for multiclass strawberry classfication.")
    parser.add_argument("--model_type", type=str, help="custom or pretrained model", default="pretrained")
    parser.add_argument("--runtime", type=str, help="inference stack e.g., [tensorflow, oonnxruntime]", default='tensorflow')
    parser.add_argument("--image_size", type=tuple, help="The image size for model input", default=(512,512))
    parser.add_argument("--batch_size", type=int, help="The batch size for model building", default=32)
    parser.add_argument("--epochs", type=int, help='Number of epochs', default=20)

    args = parser.parse_args()

    model_type = args.model_type
    runtime = args.runtime
    image_size = args.image_size
    batch_size = args.batch_size
    n_epochs = args.epochs

    # Verify the inference stack to be employed
    if runtime.lower() == 'tensorflow': 
        model_path = os.path.join(models_directory, model_type + '.keras')
    elif runtime.lower == 'onnxruntime':
        model_path = os.path.join(models_directory, model_type + '.onnx')
    else:
        print(f"Invalid runtime: {runtime.lower()}")
        exit(1)

    # Check if model already exists; if not, build and train model
    if not os.path.exists(model_path):
        
        print("Loading datasets.")
        # Load the training and validation sets
        train_dataset, val_dataset, test_dataset, class_names = load_datasets(dataset_directory, image_size, batch_size)
        
        # Print dataset information
        print(f"Image size: {image_size}")
        print(f"Batch size: {batch_size}")
        print(f"Number of epochs: {n_epochs}")
        print(f"Number of batches in train dataset: {len(train_dataset)}")
        print(f"Number of batches in validation dataset: {len(val_dataset)}")
        print(f"Number of batches in test dataset: {len(test_dataset)}")
        print(f"Class names: {class_names}")
        
        # Display an image from the dataset
        image, variety = process_image(dataset_directory, class_names)
        display_image(image, variety)

        print("Building model.")

        if model_type.lower() == "custom":
            # Build custom CNN model
            model = custom_model(image_size, len(class_names))
            # Print model summary
            model.summary()
            
        else:
            # Build pretrained ImageNet model
            model = pretrained_model(image_size, len(class_names))
            # Print model summary
            model.summary()

        print(f"Training model.")
        # Train model with the optimal hyperparameter set
        start_time = time.perf_counter()
        history = train(model, model_path, train_dataset, val_dataset, n_epochs, batch_size)
        end_time = time.perf_counter()
        print(f"Time to train model: {end_time - start_time}")

        # Plot accuracy and loss curves
        fig_path = os.path.join(assets_directory, model_type)
        plot_curves(history, fig_path)

        print("Evaluating model performance on testing set.")
        evaluate(model, model_type, test_dataset)
        
        print("Saving model to the ONNX format.")
        conv_to_onnx(model, model_type)
        
    else:
        
        image_path = r"D:\Umer Project - Strawberry Identification App\test_images\Radiance.jpg"
        variety = 'Radiance'
        
        # Load model from file
        loaded_model, model_type = load_model(model_path)
        
        # Perfrom inference
        label_predicted = predict(loaded_model, model_type, image_path, variety)

        # Invoke OpenAI's LLM for prompt reponse related to the strawberry variety predicted
        # invokeLLM(label_predicted)
        
if __name__ == '__main__':

    main()

