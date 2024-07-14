import os

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
