import requests
import time
import json
from datetime import datetime


def test_prometheus_connection():
    """Test if Prometheus is running and accessible"""
    try:
        response = requests.get('http://localhost:9090/-/healthy')
        if response.status_code == 200:
            print("Prometheus health check: SUCCESS")
            return True
        else:
            print(
                f"Prometheus health check: FAILED (status code {response.status_code})")
            return False
    except Exception as e:
        print(f"Prometheus connection error: {e}")
        return False


def test_prometheus_metrics():
    """Test if Prometheus is collecting metrics"""
    try:
        # Query for up metric (shows which targets are up)
        response = requests.get(
            'http://localhost:9090/api/v1/query', params={'query': 'up'})

        if response.status_code != 200:
            print(
                f"Prometheus metrics query failed: status code {response.status_code}")
            return False

        data = response.json()
        if data['status'] != 'success':
            print(f"Prometheus query returned error: {data['status']}")
            return False

        results = data['data']['result']
        print(f"\nFound {len(results)} monitored targets:")

        for result in results:
            instance = result['metric'].get('instance', 'unknown')
            job = result['metric'].get('job', 'unknown')
            up = int(result['value'][1])
            status = "UP" if up == 1 else "DOWN"
            print(f"  - {job} ({instance}): {status}")

        return True
    except Exception as e:
        print(f"Error querying Prometheus metrics: {e}")
        return False


def test_prometheus_query_range():
    """Test Prometheus range queries"""
    try:
        # Get CPU usage over the last 5 minutes
        end_time = int(time.time())
        start_time = end_time - 300  # 5 minutes ago
        step = '15s'  # 15-second resolution

        response = requests.get(
            'http://localhost:9090/api/v1/query_range',
            params={
                'query': 'rate(process_cpu_seconds_total[1m])',
                'start': start_time,
                'end': end_time,
                'step': step
            }
        )

        if response.status_code != 200:
            print(
                f"Prometheus range query failed: status code {response.status_code}")
            return False

        data = response.json()
        if data['status'] != 'success':
            print(f"Prometheus range query returned error: {data['status']}")
            return False

        results = data['data']['result']
        if results:
            print(
                f"\nSuccessfully queried CPU usage data for {len(results)} series")
            for result in results:
                instance = result['metric'].get('instance', 'unknown')
                job = result['metric'].get('job', 'unknown')
                datapoints = len(result['values'])
                print(f"  - {job} ({instance}): {datapoints} datapoints")
            return True
        else:
            print(
                "No CPU usage data found. This might be normal for a newly started system.")
            return True
    except Exception as e:
        print(f"Error querying Prometheus range data: {e}")
        return False


def test_prometheus_alerts():
    """Test if Prometheus alerts are configured"""
    try:
        response = requests.get('http://localhost:9090/api/v1/alerts')

        if response.status_code != 200:
            print(
                f"Prometheus alerts query failed: status code {response.status_code}")
            return False

        data = response.json()
        if data['status'] != 'success':
            print(f"Prometheus alerts query returned error: {data['status']}")
            return False

        alerts = data['data']['alerts']
        print(f"\nFound {len(alerts)} configured alerts")

        return True
    except Exception as e:
        print(f"Error querying Prometheus alerts: {e}")
        return False


if __name__ == "__main__":
    print("=== Prometheus Test ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if test_prometheus_connection():
        test_prometheus_metrics()
        test_prometheus_query_range()
        test_prometheus_alerts()

    print("=== Test Complete ===")
