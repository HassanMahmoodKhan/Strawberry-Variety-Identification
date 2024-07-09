from PIL import Image
import random
import os

def check_images(directory):
    corrupted_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img.verify()  # Verify that it is a valid image
                except (IOError, SyntaxError) as e:
                    print(f"Corrupted file: {file_path}")
                    corrupted_files.append(file_path)
    return corrupted_files

def remove_random_images(directory, remove_fraction=0.1):
    """
    Remove a fraction of random images from the directory.

    Parameters:
    directory (str): Path to the dataset directory.
    remove_fraction (float): Fraction of images to remove (between 0 and 1).
    """
    # Collect all image file paths
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                image_files.append(os.path.join(root, file))
    
    # Determine number of images to remove
    num_images = len(image_files)
    num_to_remove = int(num_images * remove_fraction)
    
    # Shuffle and select images to remove
    random.shuffle(image_files)
    files_to_remove = image_files[:num_to_remove]
    
    # Remove selected images
    for file_path in files_to_remove:
        os.remove(file_path)
    
    print(f"Total images before file removal: {num_images}")
    print(f"Removing {num_to_remove} images")
    print(f"Total images after file removal: {num_images - num_to_remove}")

if __name__ == '__main__':

    # Dataset directory
    image_directory = os.path.join(os.getcwd(), 'dataset')
    remove_random_images(image_directory, 0.5)

    # corrupted_files = check_images(image_directory)
    # print(f"Found {len(corrupted_files)} corrupted files.")

    # for file_path in corrupted_files:
    #     os.remove(file_path)
    #     print(f"Removed corrupted file: {file_path}")