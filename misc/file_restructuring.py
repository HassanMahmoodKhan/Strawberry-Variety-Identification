import os
import shutil 

# Set new working directory
current_directory = os.getcwd()
print(f"Present working directory: {current_directory}")

# Create dataset directory if it does not exist
dataset_directory = os.path.join(current_directory,'dataset')
if not os.path.exists(dataset_directory):
    os.mkdir(dataset_directory)
    print(f"Dataset directory: {dataset_directory}")


def is_image(file_name):
    """
    Checks if a file is an image, returns True else False
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
    return file_name.lower().endswith(image_extensions)


def create_directory_structure(labels):
    """
    Create subdirectories for each strawberry variety within the /dataset parent directory
    """
    for label in labels:
        strawberry_type_path = os.path.join(dataset_directory, label)
        if not os.path.exists(strawberry_type_path):
            os.mkdir(strawberry_type_path)
            print(strawberry_type_path)
        else:
            continue


def restructure_directory(paths):
    """
    Reorganizes the original image dataset in a predefined directory structure for ease of access.
    """
    for path in paths:
        dir = os.path.join(current_directory, path)
        # Access each strawberry variety directory
        for folder in os.listdir(dir):
            src_directory = os.path.join(dir, folder)
            if not os.path.isdir(src_directory):
                print("The provided path is not a valid directory")
            else:
                for file_name in os.listdir(src_directory):
                    src_file = os.path.join(src_directory, file_name)
                        
                    # Check if file is an image
                    if is_image(file_name):
                        label = file_name.split('_')[0]
                        dst_directory = os.path.join(dataset_directory, label)
                        
                        # Check if destination directory exists, if not create it
                        if not os.path.exists(dst_directory):
                            os.makedirs(dst_directory)

                        # Copy image from source to destination    
                        dst_file = os.path.join(dst_directory, file_name)
                        shutil.copy(src_file, dst_file)
                        print(f"Successfully copied {file_name} to {dst_directory}")
    

if __name__=='__main__':

    paths = ['Pictures_01/pictures_1_269', 'Pictures_02/2_1975', 'Pictures_03/3_SanAndreas',
            'Pictures_04/4_Radiance', 'Pictures_05/5_Monterey', 'Pictures_06/6_Benecia',
            'Pictures_07/7_Fortuna']
    labels = ['269', '1975', 'SanAndreas', 'Radiance', 'Monterey', 'Benadice', 'Fortuna']
    
    create_directory_structure(labels)

    # restructure_directory(paths)