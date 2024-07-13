import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def load_datasets(dataset_directory, image_size, batch_size):
    """
    Load the training and validation datasets, with prefetching.

    Parameters:
        dataset_directory: The path to the dataset directory.
        image_size: The expected image shape/size for the dataset i.e., tuple(x,y)
        batch_size: The expected batch size for the dataset.

    Returns:
        train_dataset: TensorFlow train dataset object.
        validation_dataset: TensorFlow validation dataset object.
        test_dataset: TensorFlow test datset object.
        class_names: The dataset labels.
    """

    # Loads images from the dataset directory, labels according to the subdirectory names, and returns a tf.data.Dataset object
    train_dataset = image_dataset_from_directory(
            dataset_directory,
            validation_split=0.3,
            subset='training',
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='categorical'  # One-hot encoded labels
    )

    temp_validation_dataset = image_dataset_from_directory(
            dataset_directory,
            validation_split=0.3,
            subset='validation',
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='categorical'
    )
    # Splitting the dataset
    validation_size = int(0.5 * len(temp_validation_dataset))  # 15% for validation
    test_size = len(temp_validation_dataset) - validation_size  # 15% for testing

    validation_dataset = temp_validation_dataset.take(validation_size)
    test_dataset = temp_validation_dataset.skip(validation_size).take(test_size)

    class_names = train_dataset.class_names
    print(class_names)

    # Prefetch and cache the datasets for better performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset, class_names