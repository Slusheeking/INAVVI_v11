"""
Performance Metrics Collection for Autonomous Trading System

This module provides comprehensive metrics collection for the autonomous trading system,
tracking API response times, data processing times, model training and inference times,
trade execution times, model accuracy, data quality, and system latency.
"""

import logging
import time

class PerformanceMetrics:
    """Collects and reports performance metrics for the trading system."""
    
    def __init__(self):
        self.metrics = {
            "api_response_times": {},
            "data_processing_times": {},
            "model_training_times": {},
            "model_inference_times": {},
            "trade_execution_times": {},
            "model_accuracy": {},
            "data_quality": {},
            "system_latency": {}
        }
        self.logger = logging.getLogger("performance_metrics")
        
    def record_api_response_time(self, api_name, endpoint, response_time):
        """Record API response time."""
        if api_name not in self.metrics["api_response_times"]:
            self.metrics["api_response_times"][api_name] = {}
            
        if endpoint not in self.metrics["api_response_times"][api_name]:
            self.metrics["api_response_times"][api_name][endpoint] = []
            
        self.metrics["api_response_times"][api_name][endpoint].append(response_time)
        
    def record_data_processing_time(self, operation, processing_time):
        """Record data processing time."""
        if operation not in self.metrics["data_processing_times"]:
            self.metrics["data_processing_times"][operation] = []
            
        self.metrics["data_processing_times"][operation].append(processing_time)
        
    def record_model_training_time(self, model_type, training_time):
        """Record model training time."""
        if model_type not in self.metrics["model_training_times"]:
            self.metrics["model_training_times"][model_type] = []
            
        self.metrics["model_training_times"][model_type].append(training_time)
        
    def record_model_inference_time(self, model_type, inference_time):
        """Record model inference time."""
        if model_type not in self.metrics["model_inference_times"]:
            self.metrics["model_inference_times"][model_type] = []
            
        self.metrics["model_inference_times"][model_type].append(inference_time)
        
    def record_trade_execution_time(self, operation, execution_time):
        """Record trade execution time."""
        if operation not in self.metrics["trade_execution_times"]:
            self.metrics["trade_execution_times"][operation] = []
            
        self.metrics["trade_execution_times"][operation].append(execution_time)
        
    def record_model_accuracy(self, model_type, accuracy):
        """Record model accuracy."""
        if model_type not in self.metrics["model_accuracy"]:
            self.metrics["model_accuracy"][model_type] = []
            
        self.metrics["model_accuracy"][model_type].append(accuracy)
        
    def record_data_quality(self, data_type, quality_score):
        """Record data quality score."""
        if data_type not in self.metrics["data_quality"]:
            self.metrics["data_quality"][data_type] = []
            
        self.metrics["data_quality"][data_type].append(quality_score)
        
    def record_system_latency(self, component, latency):
        """Record system latency."""
        if component not in self.metrics["system_latency"]:
            self.metrics["system_latency"][component] = []
            
        self.metrics["system_latency"][component].append(latency)
        
    def get_api_response_time_stats(self, api_name=None, endpoint=None):
        """Get API response time statistics."""
        if api_name:
            if api_name not in self.metrics["api_response_times"]:
                return {}
                
            if endpoint:
                if endpoint not in self.metrics["api_response_times"][api_name]:
                    return {}
                    
                times = self.metrics["api_response_times"][api_name][endpoint]
                return self._calculate_stats(times)
                
            # Calculate stats for all endpoints of the specified API
            stats = {}
            for ep, times in self.metrics["api_response_times"][api_name].items():
                stats[ep] = self._calculate_stats(times)
            return stats
            
        # Calculate stats for all APIs
        stats = {}
        for api, endpoints in self.metrics["api_response_times"].items():
            api_stats = {}
            for ep, times in endpoints.items():
                api_stats[ep] = self._calculate_stats(times)
            stats[api] = api_stats
        return stats
        
    def get_data_processing_time_stats(self, operation=None):
        """Get data processing time statistics."""
        if operation:
            if operation not in self.metrics["data_processing_times"]:
                return {}
                
            times = self.metrics["data_processing_times"][operation]
            return self._calculate_stats(times)
            
        # Calculate stats for all operations
        stats = {}
        for op, times in self.metrics["data_processing_times"].items():
            stats[op] = self._calculate_stats(times)
        return stats
        
    def get_model_training_time_stats(self, model_type=None):
        """Get model training time statistics."""
        if model_type:
            if model_type not in self.metrics["model_training_times"]:
                return {}
                
            times = self.metrics["model_training_times"][model_type]
            return self._calculate_stats(times)
            
        # Calculate stats for all model types
        stats = {}
        for mt, times in self.metrics["model_training_times"].items():
            stats[mt] = self._calculate_stats(times)
        return stats
        
    def get_model_inference_time_stats(self, model_type=None):
        """Get model inference time statistics."""
        if model_type:
            if model_type not in self.metrics["model_inference_times"]:
                return {}
                
            times = self.metrics["model_inference_times"][model_type]
            return self._calculate_stats(times)
            
        # Calculate stats for all model types
        stats = {}
        for mt, times in self.metrics["model_inference_times"].items():
            stats[mt] = self._calculate_stats(times)
        return stats
        
    def get_trade_execution_time_stats(self, operation=None):
        """Get trade execution time statistics."""
        if operation:
            if operation not in self.metrics["trade_execution_times"]:
                return {}
                
            times = self.metrics["trade_execution_times"][operation]
            return self._calculate_stats(times)
            
        # Calculate stats for all operations
        stats = {}
        for op, times in self.metrics["trade_execution_times"].items():
            stats[op] = self._calculate_stats(times)
        return stats
        
    def get_model_accuracy_stats(self, model_type=None):
        """Get model accuracy statistics."""
        if model_type:
            if model_type not in self.metrics["model_accuracy"]:
                return {}
                
            accuracies = self.metrics["model_accuracy"][model_type]
            return self._calculate_stats(accuracies)
            
        # Calculate stats for all model types
        stats = {}
        for mt, accuracies in self.metrics["model_accuracy"].items():
            stats[mt] = self._calculate_stats(accuracies)
        return stats
        
    def get_data_quality_stats(self, data_type=None):
        """Get data quality statistics."""
        if data_type:
            if data_type not in self.metrics["data_quality"]:
                return {}
                
            scores = self.metrics["data_quality"][data_type]
            return self._calculate_stats(scores)
            
        # Calculate stats for all data types
        stats = {}
        for dt, scores in self.metrics["data_quality"].items():
            stats[dt] = self._calculate_stats(scores)
        return stats
        
    def get_system_latency_stats(self, component=None):
        """Get system latency statistics."""
        if component:
            if component not in self.metrics["system_latency"]:
                return {}
                
            latencies = self.metrics["system_latency"][component]
            return self._calculate_stats(latencies)
            
        # Calculate stats for all components
        stats = {}
        for comp, latencies in self.metrics["system_latency"].items():
            stats[comp] = self._calculate_stats(latencies)
        return stats
        
    def _calculate_stats(self, values):
        """Calculate statistics for a list of values."""
        if not values:
            return {}
            
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2],
            "p95": sorted(values)[int(len(values) * 0.95)] if len(values) >= 20 else max(values),
            "p99": sorted(values)[int(len(values) * 0.99)] if len(values) >= 100 else max(values)
        }
        
    def generate_report(self):
        """Generate a comprehensive performance report."""
        report = {
            "api_response_times": self.get_api_response_time_stats(),
            "data_processing_times": self.get_data_processing_time_stats(),
            "model_training_times": self.get_model_training_time_stats(),
            "model_inference_times": self.get_model_inference_time_stats(),
            "trade_execution_times": self.get_trade_execution_time_stats(),
            "model_accuracy": self.get_model_accuracy_stats(),
            "data_quality": self.get_data_quality_stats(),
            "system_latency": self.get_system_latency_stats()
        }
        
        return report
        
    def log_report(self):
        """Log a summary of the performance report."""
        report = self.generate_report()
        
        self.logger.info("=== Performance Report ===")
        
        # Log API response times
        self.logger.info("API Response Times:")
        for api, endpoints in report["api_response_times"].items():
            for endpoint, stats in endpoints.items():
                self.logger.info(f"  {api} - {endpoint}: mean={stats['mean']:.3f}s, p95={stats.get('p95', stats['max']):.3f}s")
                
        # Log model accuracy
        self.logger.info("Model Accuracy:")
        for model_type, stats in report["model_accuracy"].items():
            if stats:
                self.logger.info(f"  {model_type}: mean={stats['mean']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}")
                
        # Log system latency
        self.logger.info("System Latency:")
        for component, stats in report["system_latency"].items():
            if stats:
                self.logger.info(f"  {component}: mean={stats['mean']:.3f}s, p95={stats.get('p95', stats['max']):.3f}s")
                
        return report