import onnx
import tensorflow as tf
import tf2onnx

def conv_to_onnx(model, model_type):
    """"
    The function serializes the Keras model to its ONNX equivalent.

    Parameters:
        model: The Keras model object
        model_type: The type of model i.e., custom, pretrained, etc.

    Returns:
        None
    """
    # Define the input signature of the keras model
    input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='input')]

    # Convert the Keras model to its ONNX equivalent
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature,
        opset=14,
        output_path=f'models/{model_type}.onnx'
    )

    # Print a readable version of the graph
    print("Model conversion successful!")
    print(onnx.helper.printable_graph(onnx_model.graph))