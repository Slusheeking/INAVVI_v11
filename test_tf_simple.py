import os
import sys

# Try to make TensorFlow less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    print('Attempting to import TensorFlow...')
    import tensorflow as tf
    print(f'TensorFlow version: {tf.__version__}')
    
    # Check for GPU devices
    print('\nChecking for GPU devices:')
    physical_devices = tf.config.list_physical_devices()
    for device in physical_devices:
        print(f'  {device.device_type}: {device.name}')
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f'\nNumber of GPU devices: {len(gpu_devices)}')
    
    if len(gpu_devices) > 0:
        print('GPU is available for TensorFlow!')
    else:
        print('No GPU available for TensorFlow.')
        
except Exception as e:
    print(f'Error importing TensorFlow: {e}')
    print(f'Python version: {sys.version}')
    print(f'Python path: {sys.executable}')
    
    # Try to get more information about the environment
    try:
        import numpy as np
        print(f'NumPy version: {np.__version__}')
    except Exception as np_error:
        print(f'Error importing NumPy: {np_error}')
    
    try:
        import platform
        print(f'Platform: {platform.platform()}')
    except Exception:
        pass
