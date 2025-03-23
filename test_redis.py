import redis
import time
import random
import string


def random_string(length=10):
    """Generate a random string of fixed length"""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def test_redis_connection():
    """Test basic Redis connection"""
    try:
        # Use the correct Redis connection parameters from docker-compose.unified.yml
        r = redis.Redis(
            host='localhost',
            port=6380,
            db=0,
            password='trading_system_2025',
            username='default',
            decode_responses=True
        )
        ping_result = r.ping()
        print(
            f"Redis connection test: {'SUCCESS' if ping_result else 'FAILED'}")
        return r if ping_result else None
    except Exception as e:
        print(f"Redis connection error: {e}")
        return None


def test_redis_operations(r):
    """Test basic Redis operations"""
    if not r:
        return

    # Test SET and GET
    test_key = f"test:{random_string()}"
    test_value = random_string(20)

    print("\nTesting SET/GET operations...")
    start = time.time()
    r.set(test_key, test_value)
    retrieved = r.get(test_key)
    elapsed = time.time() - start

    if retrieved == test_value:
        print(f"SET/GET test: SUCCESS (completed in {elapsed*1000:.2f} ms)")
    else:
        print(
            f"SET/GET test: FAILED (got {retrieved} instead of {test_value})")

    # Test HSET and HGET
    test_hash = f"hash:{random_string()}"
    test_field = random_string(5)
    test_hash_value = random_string(20)

    print("\nTesting HSET/HGET operations...")
    start = time.time()
    r.hset(test_hash, test_field, test_hash_value)
    hash_retrieved = r.hget(test_hash, test_field)
    elapsed = time.time() - start

    if hash_retrieved == test_hash_value:
        print(f"HSET/HGET test: SUCCESS (completed in {elapsed*1000:.2f} ms)")
    else:
        print(f"HSET/HGET test: FAILED")

    # Test PIPELINE
    print("\nTesting PIPELINE operations...")
    pipeline_size = 1000
    pipeline = r.pipeline()

    start = time.time()
    for i in range(pipeline_size):
        pipeline.set(f"pipeline:{random_string()}", random_string(20))
    pipeline.execute()
    elapsed = time.time() - start

    print(
        f"PIPELINE test ({pipeline_size} operations): SUCCESS (completed in {elapsed*1000:.2f} ms)")
    print(f"Average time per operation: {elapsed*1000/pipeline_size:.3f} ms")

    # Clean up
    r.delete(test_key)
    r.delete(test_hash)


if __name__ == "__main__":
    print("=== Redis Test ===")
    r = test_redis_connection()
    if r:
        test_redis_operations(r)
    print("=== Test Complete ===")
