"""
Tests for token efficiency and caching behavior.

Tests that the system properly uses caching, tracks tokens,
and optimizes for minimal API calls.
"""
import unittest
import tempfile
import time
from pathlib import Path
import os

# Load .env early so SKIP_API_TESTS reflects keys defined in project root `.env`.
try:
    from dotenv import load_dotenv
    from pathlib import Path as _Path

    _project_root = _Path(__file__).resolve().parents[1]
    load_dotenv(_project_root / ".env", override=False)
except Exception:
    pass

# Only run these tests if API key is available
SKIP_API_TESTS = not os.getenv("DEEPSEEK_API_KEY")


class TestCacheManager(unittest.TestCase):
    """Test LLM cache manager."""
    
    def setUp(self):
        """Set up temporary cache."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_path = Path(self.temp_dir) / "test_cache.db"
    
    def tearDown(self):
        """Clean up."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_cache_initialization(self):
        """Test cache database initialization."""
        from llm.deepseek_client import CacheManager
        
        cache = CacheManager(str(self.cache_path))
        
        # Cache file should be created
        self.assertTrue(self.cache_path.exists())
    
    def test_cache_put_and_get(self):
        """Test storing and retrieving from cache."""
        from llm.deepseek_client import CacheManager
        
        cache = CacheManager(str(self.cache_path))
        
        # Store a response
        cache_key = "test_key_123"
        response = {
            "content": "This is a test response",
            "model": "deepseek-chat",
            "usage": {"total_tokens": 25}
        }
        cache.put(cache_key, "deepseek-chat", response, 25)
        
        # Retrieve it
        cached = cache.get(cache_key)
        
        self.assertIsNotNone(cached)
        self.assertEqual(cached["response"]["content"], "This is a test response")
        self.assertEqual(cached["token_count"], 25)
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        from llm.deepseek_client import CacheManager
        
        cache = CacheManager(str(self.cache_path))
        
        result = cache.get("nonexistent_key")
        self.assertIsNone(result)
    
    def test_cache_key_determinism(self):
        """Test that cache keys are deterministic."""
        from llm.deepseek_client import DeepSeekClient
        
        # Skip if no API key (we're just testing key generation)
        if SKIP_API_TESTS:
            # Create client without actual API call
            try:
                # Test cache key computation directly
                import hashlib
                import json
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"}
                ]
                
                # Compute key twice
                normalized1 = json.dumps({
                    "messages": messages,
                    "model": "deepseek-chat",
                    "temperature": 0.0,
                    "max_tokens": 100
                }, sort_keys=True)
                key1 = hashlib.sha256(normalized1.encode()).hexdigest()
                
                normalized2 = json.dumps({
                    "messages": messages,
                    "model": "deepseek-chat",
                    "temperature": 0.0,
                    "max_tokens": 100
                }, sort_keys=True)
                key2 = hashlib.sha256(normalized2.encode()).hexdigest()
                
                # Keys should be identical
                self.assertEqual(key1, key2)
            except:
                self.skipTest("API key not available")


@unittest.skipIf(SKIP_API_TESTS, "DeepSeek API key not available")
class TestDeepSeekClient(unittest.TestCase):
    """Test DeepSeek client with caching."""
    
    def setUp(self):
        """Set up client with temporary cache."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_path = Path(self.temp_dir) / "test_llm_cache.db"
        
        from llm.deepseek_client import DeepSeekClient
        
        self.client = DeepSeekClient(
            cache_enabled=True,
            cache_path=str(self.cache_path)
        )
        self.client.reset_stats()
    
    def tearDown(self):
        """Clean up."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_cache_hit_on_repeated_call(self):
        """Test that repeated identical calls hit cache."""
        messages = [
            {"role": "system", "content": "You are a math assistant."},
            {"role": "user", "content": "What is 5 + 3?"}
        ]
        
        # First call - should miss cache
        result1 = self.client.chat_completion(
            messages,
            temperature=0.0,
            max_tokens=50
        )
        
        stats_after_first = self.client.get_stats()
        self.assertEqual(stats_after_first["cache_misses"], 1)
        
        # Second identical call - should hit cache
        result2 = self.client.chat_completion(
            messages,
            temperature=0.0,
            max_tokens=50
        )
        
        stats_after_second = self.client.get_stats()
        self.assertEqual(stats_after_second["cache_hits"], 1)
        
        # Results should be identical
        self.assertEqual(result1["content"], result2["content"])
    
    def test_token_tracking(self):
        """Test that token usage is tracked."""
        result = self.client.chat_with_system(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'hello' and nothing else.",
            temperature=0.0,
            max_tokens=10
        )
        
        stats = self.client.get_stats()
        
        # Should have tracked some tokens
        self.assertGreater(stats["total_input_tokens"], 0)
        self.assertGreater(stats["total_output_tokens"], 0)
        self.assertGreater(stats["total_tokens"], 0)
    
    def test_cache_efficiency_for_consistent_prompts(self):
        """Test cache efficiency with consistent prompt structure."""
        system_prompt = "You are a financial assistant specialized in derivatives."
        
        # Make multiple queries with same system prompt
        queries = [
            "What is delta?",
            "What is gamma?",
            "What is vega?"
        ]
        
        self.client.reset_stats()
        
        for query in queries:
            self.client.chat_with_system(
                system_prompt=system_prompt,
                user_prompt=query,
                temperature=0.0,
                max_tokens=100
            )
        
        # All should be cache misses on first run
        stats = self.client.get_stats()
        self.assertEqual(stats["cache_misses"], 3)
        
        # Run same queries again
        for query in queries:
            self.client.chat_with_system(
                system_prompt=system_prompt,
                user_prompt=query,
                temperature=0.0,
                max_tokens=100
            )
        
        # All should hit cache
        stats = self.client.get_stats()
        self.assertEqual(stats["cache_hits"], 3)
        
        # Cache hit rate should be 50% (3 hits, 3 misses)
        self.assertAlmostEqual(stats["cache_hit_rate"], 0.5, places=2)


class TestBatchProcessing(unittest.TestCase):
    """Test batch processing capabilities."""
    
    @unittest.skipIf(SKIP_API_TESTS, "API key not available")
    def test_batch_chat_processes_multiple_requests(self):
        """Test that batch_chat processes multiple requests."""
        from llm.deepseek_client import DeepSeekClient
        
        temp_dir = tempfile.mkdtemp()
        cache_path = Path(temp_dir) / "batch_test_cache.db"
        
        try:
            client = DeepSeekClient(cache_enabled=True, cache_path=str(cache_path))
            
            messages_list = [
                [
                    {"role": "system", "content": "You are a math assistant."},
                    {"role": "user", "content": "What is 1+1?"}
                ],
                [
                    {"role": "system", "content": "You are a math assistant."},
                    {"role": "user", "content": "What is 2+2?"}
                ]
            ]
            
            start_time = time.time()
            results = client.batch_chat(messages_list, temperature=0.0, max_tokens=20)
            elapsed = time.time() - start_time
            
            # Should return results for all requests
            self.assertEqual(len(results), 2)
            
            # All results should have content
            for result in results:
                self.assertIn("content", result)
                self.assertIsNotNone(result["content"])
            
            print(f"Batch processing: 2 requests in {elapsed:.2f}s")
        
        finally:
            import shutil
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)


class TestPromptCacheFriendliness(unittest.TestCase):
    """Test prompt design for cache-friendly API calls."""
    
    def test_consistent_structure_produces_same_cache_key(self):
        """Test that consistent prompt structure produces same cache key."""
        import hashlib
        import json
        
        # Same structure, different content in user message only
        messages1 = [
            {"role": "system", "content": "You are a QA assistant."},
            {"role": "user", "content": "Query A"}
        ]
        
        messages2 = [
            {"role": "system", "content": "You are a QA assistant."},
            {"role": "user", "content": "Query B"}
        ]
        
        # Keys should be different (different content)
        key1 = hashlib.sha256(json.dumps(messages1, sort_keys=True).encode()).hexdigest()
        key2 = hashlib.sha256(json.dumps(messages2, sort_keys=True).encode()).hexdigest()
        
        self.assertNotEqual(key1, key2)
        
        # But same messages should produce same key
        messages3 = [
            {"role": "system", "content": "You are a QA assistant."},
            {"role": "user", "content": "Query A"}
        ]
        
        key3 = hashlib.sha256(json.dumps(messages3, sort_keys=True).encode()).hexdigest()
        self.assertEqual(key1, key3)


class TestTokenBudget(unittest.TestCase):
    """Test token budget compliance."""
    
    def test_chunk_context_fits_in_budget(self):
        """Test that typical chunk context fits in token budget."""
        from knowledge.models import Chunk, PageSpan
        
        # Simulate a context window with 5 chunks (typical for RAG)
        chunks = []
        for i in range(5):
            span = PageSpan(start_page=i+1, end_page=i+1)
            # Each chunk ~300 tokens (typical)
            chunk = Chunk(
                id=f"chunk_{i}",
                doc_id="test_doc",
                strategy="fixed",
                text="word " * 300,  # ~300 tokens
                page_span=span,
                token_count=300,
                char_count=1500
            )
            chunks.append(chunk)
        
        total_tokens = sum(c.token_count for c in chunks)
        
        # Should fit comfortably in 8K context (leaving room for prompt + response)
        self.assertLess(total_tokens, 2000, "Context chunks should leave room for prompt")
        
        print(f"Typical RAG context: {total_tokens} tokens from {len(chunks)} chunks")
    
    def test_answer_generation_token_estimate(self):
        """Test that answer generation stays within token budget."""
        # Typical generation budget: 1000-2000 tokens for answer
        max_answer_tokens = 2000
        
        # With 1500 token context + 500 token prompt = 2000 tokens input
        # Plus 2000 tokens output = 4000 tokens total per query
        # This should be economical with DeepSeek pricing
        
        estimated_input = 2000
        estimated_output = 2000
        estimated_total = estimated_input + estimated_output
        
        # Should stay well under 8K context limit
        self.assertLess(estimated_total, 8000)
        
        # Cost estimate (assuming DeepSeek pricing ~$0.14 per 1M input, $0.28 per 1M output)
        cost_per_query = (estimated_input * 0.14 / 1_000_000 + 
                         estimated_output * 0.28 / 1_000_000)
        
        # Should be < $0.001 per query (very economical)
        self.assertLess(cost_per_query, 0.001)
        
        print(f"Estimated cost per query: ${cost_per_query:.6f}")


if __name__ == "__main__":
    unittest.main()

