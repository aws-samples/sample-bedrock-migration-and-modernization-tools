"""Token bucket rate limiter for per-model RPM control."""

import time
import threading
from collections import defaultdict
from datetime import datetime


class TokenBucketRateLimiter:
    """
    Thread-safe token bucket rate limiter for controlling requests per minute (RPM).

    Each model-region combination gets its own bucket to independently control
    request rates for reliability and throttling testing.
    """

    # Seconds of request burst allowed before steady-state throttling kicks in.
    # Keeps the average rate at target_rpm while letting the parallel_calls pool
    # fire concurrently instead of being serialized into a 1-at-a-time drip.
    BURST_SECONDS = 5.0

    def __init__(self):
        """Initialize the rate limiter with empty buckets."""
        self.buckets = {}  # key: "model_id@region" -> bucket state
        self.metrics = defaultdict(lambda: {
            "throttle_count": 0,
            "total_wait_time": 0,
            "request_timestamps": [],
            "actual_rpm": 0
        })
        self.lock = threading.Lock()

    def _get_bucket_key(self, model_id, region):
        """Generate unique key for model-region combination."""
        return f"{model_id}@{region}"

    def acquire(self, model_id, region, target_rpm):
        """
        Acquire permission to make a request, blocking if necessary to maintain target RPM.

        Args:
            model_id: Model identifier
            region: AWS region
            target_rpm: Target requests per minute (None = no rate limiting)

        Returns:
            dict: Metrics about this acquisition (throttled, wait_time)
        """
        if target_rpm is None or target_rpm <= 0:
            # No rate limiting
            return {"throttled": False, "wait_time": 0}

        bucket_key = self._get_bucket_key(model_id, region)

        with self.lock:
            current_time = time.time()

            # Initialize bucket if it doesn't exist
            if bucket_key not in self.buckets:
                # Capacity = burst allowance. The old value (target_rpm/60) was < 1 for
                # any target_rpm <= 60, so tokens never reached 1.0 and EVERY acquire fell
                # into the sleep branch — serializing the whole parallel_calls pool into a
                # 1-req-at-a-time drip. Allow a few seconds of burst so threads can fire
                # together up to capacity; average rate is still bounded by the refill rate.
                rate_per_sec = target_rpm / 60.0
                capacity = max(1.0, rate_per_sec * self.BURST_SECONDS)
                self.buckets[bucket_key] = {
                    "tokens": capacity,  # start full so an initial burst is allowed
                    "last_update": current_time,
                    "target_rpm": target_rpm,
                    "capacity": capacity,
                }

            bucket = self.buckets[bucket_key]

            # Refill tokens based on time elapsed
            time_elapsed = current_time - bucket["last_update"]
            tokens_to_add = time_elapsed * (target_rpm / 60.0)  # tokens per second
            bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + tokens_to_add)
            bucket["last_update"] = current_time

            # Check if we have tokens available
            if bucket["tokens"] >= 1.0:
                # We have tokens, consume one
                bucket["tokens"] -= 1.0

                # Track request timestamp
                self.metrics[bucket_key]["request_timestamps"].append(current_time)

                return {"throttled": False, "wait_time": 0}
            else:
                # Need to wait for tokens
                tokens_needed = 1.0 - bucket["tokens"]
                wait_time = tokens_needed / (target_rpm / 60.0)  # seconds to wait

                # Track throttle event
                self.metrics[bucket_key]["throttle_count"] += 1
                self.metrics[bucket_key]["total_wait_time"] += wait_time

        # Sleep outside the lock to allow other threads to proceed
        time.sleep(wait_time)

        # After waiting, try to acquire again
        with self.lock:
            current_time = time.time()
            bucket = self.buckets[bucket_key]

            # Refill tokens based on wait time
            time_elapsed = current_time - bucket["last_update"]
            tokens_to_add = time_elapsed * (target_rpm / 60.0)
            bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + tokens_to_add)
            bucket["last_update"] = current_time

            # Consume token
            bucket["tokens"] -= 1.0

            # Track request timestamp
            self.metrics[bucket_key]["request_timestamps"].append(current_time)

            return {"throttled": True, "wait_time": wait_time}

    def get_metrics(self, model_id, region):
        """
        Get metrics for a specific model-region combination.

        Args:
            model_id: Model identifier
            region: AWS region

        Returns:
            dict: Metrics including throttle_count, total_wait_time, actual_rpm
        """
        bucket_key = self._get_bucket_key(model_id, region)

        with self.lock:
            if bucket_key not in self.metrics:
                return {
                    "throttle_count": 0,
                    "total_wait_time": 0,
                    "actual_rpm": 0,
                    "target_rpm": None
                }

            metrics = self.metrics[bucket_key].copy()

            # Calculate actual RPM from request timestamps
            timestamps = metrics["request_timestamps"]
            if len(timestamps) >= 2:
                # Get timestamps from last minute
                current_time = time.time()
                recent_timestamps = [ts for ts in timestamps if current_time - ts <= 60]
                metrics["actual_rpm"] = len(recent_timestamps)
            else:
                metrics["actual_rpm"] = 0

            # Add target RPM from bucket
            if bucket_key in self.buckets:
                metrics["target_rpm"] = self.buckets[bucket_key]["target_rpm"]
            else:
                metrics["target_rpm"] = None

            # Remove timestamps from returned metrics (too large)
            del metrics["request_timestamps"]

            return metrics

    def reset_metrics(self, model_id=None, region=None):
        """
        Reset metrics for a specific model-region or all models.

        Args:
            model_id: Model identifier (None = reset all)
            region: AWS region (None = reset all)
        """
        with self.lock:
            if model_id is None and region is None:
                # Reset all metrics
                self.metrics.clear()
            else:
                # Reset specific model-region
                bucket_key = self._get_bucket_key(model_id, region)
                if bucket_key in self.metrics:
                    self.metrics[bucket_key] = {
                        "throttle_count": 0,
                        "total_wait_time": 0,
                        "request_timestamps": [],
                        "actual_rpm": 0
                    }

    def get_all_metrics(self):
        """
        Get metrics for all model-region combinations.

        Returns:
            dict: Metrics keyed by "model_id@region"
        """
        with self.lock:
            result = {}
            for bucket_key in self.metrics:
                metrics = self.metrics[bucket_key].copy()

                # Calculate actual RPM
                timestamps = metrics["request_timestamps"]
                if len(timestamps) >= 2:
                    current_time = time.time()
                    recent_timestamps = [ts for ts in timestamps if current_time - ts <= 60]
                    metrics["actual_rpm"] = len(recent_timestamps)
                else:
                    metrics["actual_rpm"] = 0

                # Add target RPM
                if bucket_key in self.buckets:
                    metrics["target_rpm"] = self.buckets[bucket_key]["target_rpm"]
                else:
                    metrics["target_rpm"] = None

                # Remove timestamps
                del metrics["request_timestamps"]

                result[bucket_key] = metrics

            return result
