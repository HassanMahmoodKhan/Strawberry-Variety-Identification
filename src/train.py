from tensorflow.keras.applications import VGG16 
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


def custom_model(image_shape=(512,512), num_classes=7):
    """
    This function builds the model and compiles it.

    Parameters:
        image_shape: The expected input image shape by the model.
        num_classes: The number of classes or labels.

    Returns:
        model: The compiled Keras model.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(image_shape[0], image_shape[1], 3), name='input'))      # Input layer
    model.add(layers.Rescaling(1/255.0))                                                  # Rescaling layer
    model.add(layers.Conv2D(filters= 32, kernel_size= (3,3), activation='relu'))          # Covolutional layer
    model.add(layers.MaxPooling2D((2,2)))                                                 # Maxpooling layer
    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3,5),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())                                                           # Flattening layer
    model.add(layers.Dense(128, activation='relu'))                                       # Fully Connected layer
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))                                                        # Dropout layer
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))             # Output layer
  
    model.output_names=['output']

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model

def pretrained_model(image_shape=(512,512), num_classes=7):
    """"
    This function loads a pretrained model e.g., ImageNet, and employs transfer learning to train 
    the top layers (custom) for the strawberry classfication task.
    """
    # Load the pretrained VGG16 model without the top layers 
    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(image_shape[0], image_shape[1], 3))
    
    # Add custom (trainable) layers on top of the base model
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes, activation='softmax', name='output')(x)

    # Create the comple model
    model = models.Model(inputs=base_model.input,
                         outputs=predictions,
                         name='VGG16' )
    
    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy'])

    return model


def train(model, model_path, train_dataset, validation_dataset, n_epochs=20, batch_size=32):
    """
    This function trains the model and saves it after each epoch, following validation accuracy improvement.
    It stops the training process if accuracy does not improve after three consecutive epochs.

    Parameters:
        model: Compiled Keras model.
        model_path: Path to save the model to.
        train_dataset: TensorFlow train dataset object.
        validation_dataset: TensorFlow validation dataset object.
        n_epochs: The number of epochs.
        batch_size: The batch size.
    
    Returns:
        history: The Keras object returned after performing model.fit().

    """
    checkpoint = callbacks.ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        verbose=1,
        mode='max',
        save_best_only=True,
        save_weights_only=False  # Save the entire model
    )

    early = callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        restore_best_weights=True,
        patience=3
    )

    callbacks_list = [checkpoint, early]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=n_epochs,
        verbose=1,
        callbacks=callbacks_list
    )

    return history
