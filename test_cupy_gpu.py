import cupy as cp
import time

def main():
    print('CuPy version:', cp.__version__)
    print('CUDA available:', cp.cuda.is_available())
    print('GPU count:', cp.cuda.runtime.getDeviceCount())
    
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

if __name__ == '__main__':
    main()
