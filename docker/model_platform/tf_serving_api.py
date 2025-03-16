import os
import tensorflow as tf
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
import numpy as np
from concurrent import futures
import time
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tensorflow_serving_api')

# Model registry path
MODEL_BASE_PATH = os.environ.get('MODEL_REGISTRY_PATH', '/app/models/registry')
MODEL_NAME = 'trading_model'


class ModelServicer(model_service_pb2_grpc.ModelServiceServicer):
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

    def GetModelStatus(self, request, context):
        response = model_management_pb2.GetModelStatusResponse()
        if request.model_spec.name in self.models:
            model_info = self.models[request.model_spec.name]
            model_version_status = response.model_version_status.add()
            model_version_status.version = model_info['version']
            model_version_status.state = model_management_pb2.ModelVersionStatus.AVAILABLE
            model_version_status.status.error_code = status_pb2.StatusProto.OK
            model_version_status.status.error_message = ''
        else:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f'Model {request.model_spec.name} not found')
        return response


class PredictionServicer(prediction_service_pb2_grpc.PredictionServiceServicer):
    def __init__(self, model_servicer):
        self.model_servicer = model_servicer

    def Predict(self, request, context):
        try:
            model_name = request.model_spec.name
            if model_name not in self.model_servicer.models:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f'Model {model_name} not found')
                return predict_pb2.PredictResponse()

            model_info = self.model_servicer.models[model_name]
            model = model_info['model']
            signatures = model_info['signatures']

            # Get the serving signature
            serving_signature = signatures.get(
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
            if serving_signature is None:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details('Model does not have a serving signature')
                return predict_pb2.PredictResponse()

            # Convert inputs from TensorProto to Tensor
            inputs = {}
            for key, tensor_proto in request.inputs.items():
                inputs[key] = tf.make_tensor_from_tensor_proto(tensor_proto)

            # Run prediction
            outputs = serving_signature(**inputs)

            # Create response
            response = predict_pb2.PredictResponse()
            response.model_spec.name = model_name
            response.model_spec.version.value = model_info['version']

            # Convert outputs to TensorProto
            for key, tensor in outputs.items():
                tf.make_tensor_proto(tensor, shape=tensor.shape).CopyFrom(
                    response.outputs[key])

            return response
        except Exception as e:
            logger.error(f'Error during prediction: {e}')
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error during prediction: {e}')
            return predict_pb2.PredictResponse()


def serve():
    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Add servicers
    model_servicer = ModelServicer()
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(
        model_servicer, server)
    prediction_servicer = PredictionServicer(model_servicer)
    prediction_service_pb2_grpc.add_PredictionServiceServicer_to_server(
        prediction_servicer, server)

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


if __name__ == '__main__':
    serve()
