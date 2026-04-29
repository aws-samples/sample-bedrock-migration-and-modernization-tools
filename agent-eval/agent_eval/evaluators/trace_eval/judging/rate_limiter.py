# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
Rate limiting implementation using token bucket algorithm.

This module provides per-judge rate limiting to prevent overwhelming
external judge APIs with too many concurrent requests.
"""

import asyncio
import time
from typing import Dict, Optional


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for controlling request rates per judge.
    
    The token bucket algorithm allows bursts up to the bucket capacity
    while maintaining an average rate over time through token refill.
    
    Attributes:
        rate_limit: Maximum requests per second per judge
        capacity: Maximum tokens in the bucket (allows bursts)
        tokens: Current available tokens per judge
        last_refill: Last refill timestamp per judge (monotonic time)
    """
    
    def __init__(
        self,
        rate_limit: Optional[float] = None,
        capacity: Optional[float] = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            rate_limit: Maximum requests per second (None = no limit)
            capacity: Maximum burst size in tokens (defaults to rate_limit)
            
        Raises:
            ValueError: If rate_limit or capacity is <= 0
        """
        # Validate rate_limit
        if rate_limit is not None:
            if rate_limit <= 0:
                raise ValueError(
                    f"rate_limit must be positive, got {rate_limit}"
                )
        
        self.rate_limit = rate_limit
        
        # Set capacity (defaults to rate_limit for 1-second burst)
        if rate_limit is not None:
            if capacity is not None:
                if capacity <= 0:
                    raise ValueError(
                        f"capacity must be positive, got {capacity}"
                    )
                self.capacity = capacity
            else:
                self.capacity = rate_limit
        else:
            self.capacity = None
        
        self.tokens: Dict[str, float] = {}
        self.last_refill: Dict[str, float] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()  # Protects _locks dict creation
    
    async def acquire(self, judge_id: str) -> None:
        """
        Acquire a token for the given judge, blocking if necessary.
        
        This method will wait until a token is available before returning.
        If no rate limit is configured, returns immediately.
        
        Uses a fair waiting strategy that releases the lock while sleeping
        to avoid head-of-line blocking.
        
        Args:
            judge_id: Identifier for the judge making the request
        """
        if self.rate_limit is None:
            return
        
        # Get or create lock for this judge (thread-safe)
        async with self._locks_lock:
            if judge_id not in self._locks:
                self._locks[judge_id] = asyncio.Lock()
        
        judge_lock = self._locks[judge_id]
        
        while True:
            async with judge_lock:
                # Initialize tokens for new judge
                if judge_id not in self.tokens:
                    self.tokens[judge_id] = self.capacity
                    self.last_refill[judge_id] = time.monotonic()
                
                # Refill tokens based on elapsed time
                self._refill_tokens(judge_id)
                
                # If we have at least one token, consume it and return
                if self.tokens[judge_id] >= 1.0:
                    self.tokens[judge_id] -= 1.0
                    return
                
                # Calculate wait time for next token
                wait_time = (1.0 - self.tokens[judge_id]) / self.rate_limit
            
            # Release lock while sleeping to avoid blocking other coroutines
            # Add small minimum to avoid busy spinning
            await asyncio.sleep(max(wait_time, 0.001))
    
    def _refill_tokens(self, judge_id: str) -> None:
        """
        Refill tokens based on elapsed time since last refill.
        
        Uses monotonic clock to avoid issues with system time adjustments.
        
        Args:
            judge_id: Identifier for the judge
        """
        now = time.monotonic()
        elapsed = now - self.last_refill[judge_id]
        
        # Add tokens based on rate and elapsed time
        tokens_to_add = elapsed * self.rate_limit
        self.tokens[judge_id] = min(
            self.capacity,
            self.tokens[judge_id] + tokens_to_add
        )
        
        self.last_refill[judge_id] = now
    
    def reset(self, judge_id: Optional[str] = None) -> None:
        """
        Reset rate limiter state.
        
        Args:
            judge_id: Specific judge to reset, or None to reset all
        """
        if judge_id:
            self.tokens.pop(judge_id, None)
            self.last_refill.pop(judge_id, None)
            self._locks.pop(judge_id, None)
        else:
            self.tokens.clear()
            self.last_refill.clear()
            self._locks.clear()
