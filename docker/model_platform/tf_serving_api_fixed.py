import os
import tensorflow as tf
import sys
import logging
import time
from concurrent import futures
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tensorflow_serving_api')

# Try to import TensorFlow Serving APIs with error handling for protobuf compatibility issues
try:
    # Monkey patch the protobuf FileDescriptor to handle the has_options attribute issue
    import google.protobuf.descriptor
    original_init = google.protobuf.descriptor.FileDescriptor.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Safely set has_options if it doesn't exist or is not writable
        try:
            self.has_options
        except (AttributeError, TypeError):
            # If has_options doesn't exist or is not writable, we'll use a dictionary to store this info
            if not hasattr(google.protobuf.descriptor.FileDescriptor, '_has_options_dict'):
                google.protobuf.descriptor.FileDescriptor._has_options_dict = {}
            google.protobuf.descriptor.FileDescriptor._has_options_dict[self.name] = True

    # Apply the monkey patch
    google.protobuf.descriptor.FileDescriptor.__init__ = patched_init

    # Also patch the has_options property getter
    original_has_options = getattr(
        google.protobuf.descriptor.FileDescriptor, 'has_options', None)
    if original_has_options is None or isinstance(original_has_options, property):
        def get_has_options(self):
            if hasattr(google.protobuf.descriptor.FileDescriptor, '_has_options_dict'):
                return google.protobuf.descriptor.FileDescriptor._has_options_dict.get(self.name, False)
            return False

        # Try to set the property, but don't fail if it's not possible
        try:
            google.protobuf.descriptor.FileDescriptor.has_options = property(
                get_has_options)
        except (AttributeError, TypeError):
            logger.warning(
                "Could not patch FileDescriptor.has_options property")

    # Now import the TensorFlow Serving APIs
    from tensorflow_serving.apis import prediction_service_pb2_grpc
    from tensorflow_serving.apis import get_model_metadata_pb2
    from tensorflow_serving.apis import predict_pb2
    from tensorflow_serving.apis import prediction_service_pb2
    from tensorflow_serving.apis import model_service_pb2_grpc
    from tensorflow_serving.apis import model_service_pb2
    from tensorflow_serving.apis import model_management_pb2
    from tensorflow_serving.config import model_server_config_pb2
    from tensorflow_serving.util import status_pb2
    from tensorflow.python.saved_model import signature_constants
    from tensorflow.python.saved_model import tag_constants
    import grpc

    logger.info("Successfully imported TensorFlow Serving APIs")
except Exception as e:
    logger.error(f"Error importing TensorFlow Serving APIs: {e}")
    # Continue with limited functionality
    prediction_service_pb2_grpc = None
    get_model_metadata_pb2 = None
    predict_pb2 = None
    prediction_service_pb2 = None
    model_service_pb2_grpc = None
    model_service_pb2 = None
    model_management_pb2 = None
    model_server_config_pb2 = None
    status_pb2 = None
    signature_constants = None
    tag_constants = None
    grpc = None

# Model registry path
MODEL_BASE_PATH = os.environ.get('MODEL_REGISTRY_PATH', '/app/models/registry')
MODEL_NAME = 'trading_model'


class ModelServicer:
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        try:
            # Find all model versions in the registry
            if not os.path.exists(MODEL_BASE_PATH):
                logger.warning(
                    f'Model base path {MODEL_BASE_PATH} does not exist')
                return

            versions = [d for d in os.listdir(MODEL_BASE_PATH) if os.path.isdir(
                os.path.join(MODEL_BASE_PATH, d))]
            if not versions:
                logger.warning(f'No model versions found in {MODEL_BASE_PATH}')
                return

            # Load the latest version
            latest_version = max([int(v)
                                 for v in versions if v.isdigit()], default=0)
            if latest_version == 0:
                logger.warning('No valid model versions found')
                return

            model_path = os.path.join(MODEL_BASE_PATH, str(latest_version))
            logger.info(f'Loading model from {model_path}')

            # Load the model
            model = tf.saved_model.load(model_path)
            self.models[MODEL_NAME] = {
                'version': latest_version,
                'model': model,
                'signatures': model.signatures
            }
            logger.info(
                f'Successfully loaded model {MODEL_NAME} version {latest_version}')
        except Exception as e:
            logger.error(f'Error loading models: {e}')


class PredictionServicer:
    def __init__(self, model_servicer):
        self.model_servicer = model_servicer

    def predict(self, inputs, model_name=MODEL_NAME):
        """
        Make a prediction using the loaded model

        Args:
            inputs: Dictionary of input tensors
            model_name: Name of the model to use

        Returns:
            Dictionary of output tensors or None if prediction fails
        """
        try:
            if model_name not in self.model_servicer.models:
                logger.error(f'Model {model_name} not found')
                return None

            model_info = self.model_servicer.models[model_name]
            model = model_info['model']
            signatures = model_info['signatures']

            # Get the serving signature
            serving_signature = signatures.get(
                'serving_default')
            if serving_signature is None:
                logger.error('Model does not have a serving signature')
                return None

            # Run prediction
            outputs = serving_signature(**inputs)
            return outputs
        except Exception as e:
            logger.error(f'Error during prediction: {e}')
            return None


def serve():
    """
    Start the TensorFlow Serving API server
    """
    # Check if we have the required modules
    if grpc is None:
        logger.error("Cannot start gRPC server due to import errors")
        # Keep the process running but don't start the server
        while True:
            time.sleep(60)
            logger.warning(
                "TensorFlow Serving API not available due to import errors")
        return

    try:
        # Create gRPC server
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        # Add servicers
        model_servicer = ModelServicer()
        model_service_pb2_grpc.add_ModelServiceServicer_to_server(
            model_servicer, server)
        prediction_servicer = PredictionServicer(model_servicer)

        # Create a wrapper class that implements the gRPC interface
        class GrpcPredictionServicer(prediction_service_pb2_grpc.PredictionServiceServicer):
            def __init__(self, prediction_servicer):
                self.prediction_servicer = prediction_servicer

            def Predict(self, request, context):
                try:
                    model_name = request.model_spec.name
                    if model_name not in self.prediction_servicer.model_servicer.models:
                        context.set_code(grpc.StatusCode.NOT_FOUND)
                        context.set_details(f'Model {model_name} not found')
                        return predict_pb2.PredictResponse()

                    # Convert inputs from TensorProto to Tensor
                    inputs = {}
                    for key, tensor_proto in request.inputs.items():
                        inputs[key] = tf.make_tensor_from_tensor_proto(
                            tensor_proto)

                    # Run prediction
                    outputs = self.prediction_servicer.predict(
                        inputs, model_name)
                    if outputs is None:
                        context.set_code(grpc.StatusCode.INTERNAL)
                        context.set_details('Error during prediction')
                        return predict_pb2.PredictResponse()

                    # Create response
                    response = predict_pb2.PredictResponse()
                    response.model_spec.name = model_name
                    response.model_spec.version.value = self.prediction_servicer.model_servicer.models[
                        model_name]['version']

                    # Convert outputs to TensorProto
                    for key, tensor in outputs.items():
                        tensor_proto = tf.make_tensor_proto(tensor)
                        response.outputs[key].CopyFrom(tensor_proto)

                    return response
                except Exception as e:
                    logger.error(f'Error during prediction: {e}')
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(f'Error during prediction: {e}')
                    return predict_pb2.PredictResponse()

        # Add the gRPC servicer
        grpc_prediction_servicer = GrpcPredictionServicer(prediction_servicer)
        prediction_service_pb2_grpc.add_PredictionServiceServicer_to_server(
            grpc_prediction_servicer, server)

        # Start server
        server.add_insecure_port('[::]:8500')
        server.start()
        logger.info('TensorFlow Serving API started on port 8500')

        try:
            while True:
                time.sleep(3600)  # Sleep for an hour
                # Reload models periodically
                model_servicer.load_models()
        except KeyboardInterrupt:
            server.stop(0)
    except Exception as e:
        logger.error(f"Error starting TensorFlow Serving API: {e}")
        # Keep the process running but log the error
        while True:
            time.sleep(60)
            logger.warning(
                f"TensorFlow Serving API not available due to error: {e}")


if __name__ == '__main__':
    serve()
