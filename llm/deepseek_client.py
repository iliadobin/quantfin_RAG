"""
DeepSeek API client with caching, batching, and retry logic.

Optimized for token efficiency:
- Prompt caching via consistent structure
- Result caching with deterministic keys
- Batch processing support
- Rate limiting and retries
"""
import os
import time
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import sqlite3
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


logger = logging.getLogger(__name__)


class CacheManager:
    """SQLite-based cache for LLM responses."""
    
    def __init__(self, cache_path: str = "data/cache/llm_cache.db"):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize cache database."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    token_count INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON llm_cache(model)")
            conn.commit()
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response."""
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute(
                "SELECT response, token_count FROM llm_cache WHERE cache_key = ?",
                (cache_key,)
            )
            row = cursor.fetchone()
            if row:
                return {"response": json.loads(row[0]), "token_count": row[1]}
        return None
    
    def put(self, cache_key: str, model: str, response: Dict[str, Any], token_count: int):
        """Store response in cache."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache (cache_key, model, response, created_at, token_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cache_key, model, json.dumps(response), time.time(), token_count)
            )
            conn.commit()
    
    def clear_old(self, days: int = 30):
        """Clear cache entries older than N days."""
        cutoff = time.time() - (days * 86400)
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("DELETE FROM llm_cache WHERE created_at < ?", (cutoff,))
            conn.commit()


class DeepSeekClient:
    """
    Client for DeepSeek API with caching and optimization.
    
    Key features:
    - Prompt caching (consistent structure for cache hits)
    - Result caching (deterministic keys)
    - Batch processing
    - Rate limiting and retries
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        default_model: str = "deepseek-chat",
        cache_enabled: bool = True,
        cache_path: str = "data/cache/llm_cache.db"
    ):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key (or from DEEPSEEK_API_KEY env var)
            base_url: API base URL
            default_model: Default model to use
            cache_enabled: Whether to use response caching
            cache_path: Path to cache database
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided and DEEPSEEK_API_KEY env var not set")
        
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.default_model = default_model
        self.cache_enabled = cache_enabled
        
        if cache_enabled:
            self.cache = CacheManager(cache_path)
        else:
            self.cache = None
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _compute_cache_key(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Compute deterministic cache key."""
        # Normalize messages to ensure consistent ordering
        normalized = json.dumps(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            sort_keys=True
        )
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call chat completion API with caching.
        
        Args:
            messages: Chat messages in OpenAI format
            model: Model to use (default: self.default_model)
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            use_cache: Whether to use cache for this request
            **kwargs: Additional API parameters
            
        Returns:
            API response dict with 'content', 'usage', etc.
        """
        model = model or self.default_model
        
        # Check cache
        cache_key = None
        if self.cache_enabled and use_cache:
            cache_key = self._compute_cache_key(messages, model, temperature, max_tokens)
            cached = self.cache.get(cache_key)
            if cached:
                self.cache_hits += 1
                logger.debug(f"Cache hit for key {cache_key[:8]}...")
                return cached["response"]
        
        # Call API
        logger.debug(f"Calling DeepSeek API: model={model}, temp={temperature}, max_tokens={max_tokens}")
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Extract result
        result = {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "finish_reason": response.choices[0].finish_reason
        }
        
        # Update token counters
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        self.cache_misses += 1
        
        # Store in cache
        if self.cache_enabled and use_cache and cache_key:
            self.cache.put(cache_key, model, result, response.usage.total_tokens)
        
        return result
    
    def chat_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Convenience method for simple system + user chat.
        
        Optimized for cache-friendly prompts:
        - Keep system_prompt consistent across calls
        - Only vary user_prompt for related queries
        
        Returns:
            Response content string
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        result = self.chat_completion(messages, model=model, **kwargs)
        return result["content"]
    
    def batch_chat(
        self,
        messages_list: List[List[Dict[str, str]]],
        model: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple chat requests.
        
        Note: Currently processes sequentially. Can be enhanced with
        async/concurrent processing if needed.
        
        Args:
            messages_list: List of message lists
            model: Model to use
            **kwargs: Additional parameters
            
        Returns:
            List of responses
        """
        results = []
        for messages in messages_list:
            result = self.chat_completion(messages, model=model, **kwargs)
            results.append(result)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get token usage and cache statistics."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0.0
            )
        }
    
    def reset_stats(self):
        """Reset token and cache statistics."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.cache_hits = 0
        self.cache_misses = 0

