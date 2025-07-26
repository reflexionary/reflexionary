"""
Tethys - Metrics Exporter

This module provides comprehensive metrics collection and export functionality
for Tethys Financial Co-Pilot. It tracks performance, usage, and system health
metrics for observability and monitoring.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from dataclasses import dataclass, asdict
from pathlib import Path

# Import Tethys configuration
from config.app_settings import BASE_DIR

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Represents a single metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    metric_type: str  # counter, gauge, histogram, summary

class MetricsCollector:
    """
    Collects and manages metrics for Tethys.
    
    This collector provides:
    - Performance metrics
    - Usage statistics
    - Error rates
    - Memory and system metrics
    - Custom business metrics
    """
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.summaries = defaultdict(list)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Metrics storage
        self.metrics_dir = BASE_DIR / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.operation_times = defaultdict(list)
        self.max_history_size = 1000
        
        logger.info("Tethys metrics collector initialized")
    
    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Value to increment by
            labels: Metric labels
        """
        with self.lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
            
            # Store metric point
            metric_point = MetricPoint(
                name=name,
                value=float(self.counters[key]),
                timestamp=datetime.now(),
                labels=labels or {},
                metric_type="counter"
            )
            self.metrics[name].append(metric_point)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Set a gauge metric value.
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Metric labels
        """
        with self.lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
            
            # Store metric point
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                metric_type="gauge"
            )
            self.metrics[name].append(metric_point)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record a histogram metric.
        
        Args:
            name: Metric name
            value: Value to record
            labels: Metric labels
        """
        with self.lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            
            # Keep only recent values
            if len(self.histograms[key]) > self.max_history_size:
                self.histograms[key] = self.histograms[key][-self.max_history_size:]
            
            # Store metric point
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                metric_type="histogram"
            )
            self.metrics[name].append(metric_point)
    
    def record_summary(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record a summary metric.
        
        Args:
            name: Metric name
            value: Value to record
            labels: Metric labels
        """
        with self.lock:
            key = self._make_key(name, labels)
            self.summaries[key].append(value)
            
            # Keep only recent values
            if len(self.summaries[key]) > self.max_history_size:
                self.summaries[key] = self.summaries[key][-self.max_history_size:]
            
            # Store metric point
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                metric_type="summary"
            )
            self.metrics[name].append(metric_point)
    
    def record_operation_time(self, operation: str, duration_ms: float, 
                            user_id: Optional[str] = None, component: Optional[str] = None):
        """
        Record operation execution time.
        
        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            user_id: User identifier (optional)
            component: Component name (optional)
        """
        labels = {}
        if user_id:
            labels["user_id"] = user_id
        if component:
            labels["component"] = component
        
        self.record_histogram(f"{operation}_duration_ms", duration_ms, labels)
        
        # Store in operation times for quick access
        with self.lock:
            self.operation_times[operation].append(duration_ms)
            if len(self.operation_times[operation]) > self.max_history_size:
                self.operation_times[operation] = self.operation_times[operation][-self.max_history_size:]
    
    def record_user_activity(self, user_id: str, activity_type: str, details: Optional[Dict[str, Any]] = None):
        """
        Record user activity metrics.
        
        Args:
            user_id: User identifier
            activity_type: Type of activity
            details: Additional activity details
        """
        labels = {"user_id": user_id, "activity_type": activity_type}
        
        self.increment_counter("user_activity_total", 1, labels)
        
        if details:
            # Record additional metrics based on activity type
            if activity_type == "query":
                self.increment_counter("user_queries_total", 1, {"user_id": user_id})
            elif activity_type == "memory_operation":
                self.increment_counter("memory_operations_total", 1, {"user_id": user_id})
            elif activity_type == "portfolio_operation":
                self.increment_counter("portfolio_operations_total", 1, {"user_id": user_id})
    
    def record_api_request(self, endpoint: str, method: str, status_code: int, 
                          duration_ms: float, user_id: Optional[str] = None):
        """
        Record API request metrics.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            status_code: Response status code
            duration_ms: Request duration in milliseconds
            user_id: User identifier (optional)
        """
        labels = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code)
        }
        if user_id:
            labels["user_id"] = user_id
        
        # Record request count
        self.increment_counter("api_requests_total", 1, labels)
        
        # Record request duration
        self.record_histogram("api_request_duration_ms", duration_ms, labels)
        
        # Record status code distribution
        self.increment_counter("api_status_codes_total", 1, {"status_code": str(status_code)})
        
        # Record error rate
        if status_code >= 400:
            self.increment_counter("api_errors_total", 1, labels)
    
    def record_memory_operation(self, operation: str, memory_type: str, 
                               user_id: str, duration_ms: float, success: bool = True):
        """
        Record memory operation metrics.
        
        Args:
            operation: Memory operation (ingest, retrieve, etc.)
            memory_type: Type of memory
            user_id: User identifier
            duration_ms: Operation duration in milliseconds
            success: Whether operation was successful
        """
        labels = {
            "operation": operation,
            "memory_type": memory_type,
            "user_id": user_id,
            "success": str(success)
        }
        
        self.increment_counter("memory_operations_total", 1, labels)
        self.record_histogram("memory_operation_duration_ms", duration_ms, labels)
        
        if not success:
            self.increment_counter("memory_operation_errors_total", 1, labels)
    
    def record_financial_operation(self, operation: str, user_id: str, 
                                 duration_ms: float, success: bool = True):
        """
        Record financial operation metrics.
        
        Args:
            operation: Financial operation
            user_id: User identifier
            duration_ms: Operation duration in milliseconds
            success: Whether operation was successful
        """
        labels = {
            "operation": operation,
            "user_id": user_id,
            "success": str(success)
        }
        
        self.increment_counter("financial_operations_total", 1, labels)
        self.record_histogram("financial_operation_duration_ms", duration_ms, labels)
        
        if not success:
            self.increment_counter("financial_operation_errors_total", 1, labels)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all collected metrics.
        
        Returns:
            Dictionary containing metrics summary
        """
        with self.lock:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "counters": {},
                "gauges": {},
                "histograms": {},
                "summaries": {},
                "operation_times": {}
            }
            
            # Aggregate counters
            for key, value in self.counters.items():
                name, labels = self._parse_key(key)
                if name not in summary["counters"]:
                    summary["counters"][name] = []
                summary["counters"][name].append({
                    "value": value,
                    "labels": labels
                })
            
            # Aggregate gauges
            for key, value in self.gauges.items():
                name, labels = self._parse_key(key)
                if name not in summary["gauges"]:
                    summary["gauges"][name] = []
                summary["gauges"][name].append({
                    "value": value,
                    "labels": labels
                })
            
            # Aggregate histograms
            for key, values in self.histograms.items():
                name, labels = self._parse_key(key)
                if name not in summary["histograms"]:
                    summary["histograms"][name] = []
                
                if values:
                    summary["histograms"][name].append({
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "labels": labels
                    })
            
            # Aggregate operation times
            for operation, times in self.operation_times.items():
                if times:
                    summary["operation_times"][operation] = {
                        "count": len(times),
                        "min": min(times),
                        "max": max(times),
                        "mean": sum(times) / len(times),
                        "p95": self._percentile(times, 95),
                        "p99": self._percentile(times, 99)
                    }
            
            return summary
    
    def export_metrics(self, format: str = "json") -> str:
        """
        Export metrics in the specified format.
        
        Args:
            format: Export format (json, prometheus)
            
        Returns:
            Exported metrics as string
        """
        if format.lower() == "json":
            return self._export_json()
        elif format.lower() == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def save_metrics(self, filename: Optional[str] = None):
        """
        Save metrics to file.
        
        Args:
            filename: Optional filename, defaults to timestamp-based name
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tethys_metrics_{timestamp}.json"
        
        filepath = self.metrics_dir / filename
        
        try:
            metrics_data = {
                "export_timestamp": datetime.now().isoformat(),
                "summary": self.get_metrics_summary(),
                "raw_metrics": self._get_raw_metrics()
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            logger.info(f"Metrics saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving metrics to {filepath}: {e}")
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a key for metric storage."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}:{label_str}"
    
    def _parse_key(self, key: str) -> tuple:
        """Parse a metric key into name and labels."""
        if ":" not in key:
            return key, {}
        
        name, label_str = key.split(":", 1)
        labels = {}
        
        for label_pair in label_str.split(","):
            if "=" in label_pair:
                k, v = label_pair.split("=", 1)
                labels[k] = v
        
        return name, labels
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _get_raw_metrics(self) -> Dict[str, Any]:
        """Get raw metrics data."""
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {k: list(v) for k, v in self.histograms.items()},
                "summaries": {k: list(v) for k, v in self.summaries.items()},
                "operation_times": {k: list(v) for k, v in self.operation_times.items()}
            }
    
    def _export_json(self) -> str:
        """Export metrics as JSON."""
        return json.dumps(self.get_metrics_summary(), indent=2, default=str)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self.lock:
            # Export counters
            for key, value in self.counters.items():
                name, labels = self._parse_key(key)
                label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                if label_str:
                    lines.append(f'{name}{{{label_str}}} {value}')
                else:
                    lines.append(f'{name} {value}')
            
            # Export gauges
            for key, value in self.gauges.items():
                name, labels = self._parse_key(key)
                label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                if label_str:
                    lines.append(f'{name}{{{label_str}}} {value}')
                else:
                    lines.append(f'{name} {value}')
            
            # Export histograms (simplified)
            for key, values in self.histograms.items():
                name, labels = self._parse_key(key)
                if values:
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                    if label_str:
                        lines.append(f'{name}_count{{{label_str}}} {len(values)}')
                        lines.append(f'{name}_sum{{{label_str}}} {sum(values)}')
                        lines.append(f'{name}_min{{{label_str}}} {min(values)}')
                        lines.append(f'{name}_max{{{label_str}}} {max(values)}')
                    else:
                        lines.append(f'{name}_count {len(values)}')
                        lines.append(f'{name}_sum {sum(values)}')
                        lines.append(f'{name}_min {min(values)}')
                        lines.append(f'{name}_max {max(values)}')
        
        return "\n".join(lines)

# Global metrics collector
metrics_collector = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector
