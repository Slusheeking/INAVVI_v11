import os
import sys
import time

def check_gpu_with_cupy():
    print('Checking GPU with CuPy...')
    try:
        import cupy as cp
        print('CuPy version:', cp.__version__)
        print('CUDA available:', cp.cuda.is_available())
        print('GPU count:', cp.cuda.runtime.getDeviceCount())
        
        if cp.cuda.is_available():
            # Get device properties
            props = cp.cuda.runtime.getDeviceProperties(0)
            print('GPU Name:', props['name'])
            print('Compute Capability:', str(props['major']) + '.' + str(props['minor']))
            print('Total Memory:', props['totalGlobalMem'] / (1024**3), 'GB')
            print('CUDA Version:', cp.cuda.runtime.runtimeGetVersion())
            
            # Simple matrix multiplication benchmark
            print('\nRunning matrix multiplication benchmark...')
            
            # Create large matrices on GPU
            size = 5000
            print(f'Matrix size: {size}x{size}')
            
            # Warm up
            a = cp.random.random((size, size), dtype=cp.float32)
            b = cp.random.random((size, size), dtype=cp.float32)
            c = cp.matmul(a, b)
            cp.cuda.Stream.null.synchronize()
            
            # Benchmark
            start_time = time.time()
            c = cp.matmul(a, b)
            cp.cuda.Stream.null.synchronize()
            elapsed = time.time() - start_time
            
            print(f'Matrix multiplication completed in {elapsed:.4f} seconds')
            print(f'Performance: {2 * size**3 / elapsed / 1e9:.2f} GFLOPS')
            
            # Clean up
            del a, b, c
            cp.get_default_memory_pool().free_all_blocks()
            
            return True
        else:
            print('No GPU available for CuPy')
            return False
    except Exception as e:
        print(f'Error using CuPy: {e}')
        return False

def check_tensorflow():
    print('\nChecking TensorFlow...')
    try:
        import tensorflow
        print('TensorFlow module attributes:', dir(tensorflow))
        
        # Try to import from tensorflow.python directly
        try:
            from tensorflow.python import pywrap_tensorflow
            print('TensorFlow build info available')
        except Exception as e:
            print(f'Error importing pywrap_tensorflow: {e}')
        
        return True
    except Exception as e:
        print(f'Error importing TensorFlow: {e}')
        return False

def main():
    print('=== GPU Status Report ===')
    print('Python version:', sys.version)
    print('Platform:', sys.platform)
    print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))
    print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH', 'Not set'))
    
    cupy_success = check_gpu_with_cupy()
    tf_success = check_tensorflow()
    
    print('\n=== Summary ===')
    print('CuPy GPU access:', 'SUCCESS' if cupy_success else 'FAILED')
    print('TensorFlow import:', 'SUCCESS' if tf_success else 'FAILED')
    
    if cupy_success:
        print('\nThe system has a working GPU that can be accessed through CuPy.')
        print('This confirms that the CUDA drivers and runtime are properly installed.')
    
    if not tf_success:
        print('\nTensorFlow is not working properly. This could be due to:')
        print('1. Incompatible versions of TensorFlow and NumPy')
        print('2. Missing CUDA libraries or incorrect paths')
        print('3. Corrupted TensorFlow installation')

if __name__ == '__main__':
    main()
