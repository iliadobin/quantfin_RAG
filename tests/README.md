# QA Assistant Test Suite

Comprehensive test suite for the quantitative finance QA assistant.

## Test Structure

### Unit Tests
- **`test_models.py`**: Data models, validation, serialization
- **`test_contracts.py`**: RAG protocol interfaces and compliance
- **`test_ingest_pipeline.py`**: Text normalization and chunking

### Integration Tests
- **`test_integration.py`**: Full pipeline integration (ingest→index→query)

### Performance Tests
- **`test_performance.py`**: Latency, throughput, memory usage

### Cache & Token Tests
- **`test_cache_tokens.py`**: Caching behavior, token tracking, efficiency

## Running Tests

### Run All Tests
```bash
python scripts/run_tests.py
```

### Run Specific Test Categories
```bash
# Unit tests only
python scripts/run_tests.py --unit

# Integration tests only
python scripts/run_tests.py --integration

# Performance tests only
python scripts/run_tests.py --performance

# Cache/token tests only
python scripts/run_tests.py --cache
```

### Fast Mode (Skip API Tests)
```bash
# Skip tests that require API calls
python scripts/run_tests.py --fast
```

### Run Individual Test Files
```bash
# Run specific test file
python -m pytest tests/test_models.py

# Or with unittest
python -m unittest tests.test_models
```

## Test Requirements

### Required
- pydantic>=2.0.0
- pyyaml>=6.0

### Optional (for full test coverage)
- DEEPSEEK_API_KEY environment variable (for cache/API tests)

## Test Coverage

The test suite covers:

1. **Data Models** (100% coverage)
   - All knowledge models
   - Validation and serialization
   - Helper methods

2. **RAG Contracts** (100% coverage)
   - Retriever protocol
   - Reranker protocol
   - Generator protocol
   - Pipeline protocol
   - Component interoperability

3. **Ingest Pipeline**
   - Text normalization
   - Fixed chunking
   - Section-aware chunking
   - Multi-page handling
   - Overlap strategies

4. **Integration**
   - Parse → normalize → chunk flow
   - Document traceability
   - Citation construction
   - Answer generation

5. **Performance**
   - Chunking latency (< 1s for 100 pages)
   - Normalization throughput (> 10 MB/s)
   - Memory usage (< 50MB for typical workloads)
   - Retrieval operations (< 0.1s for 1000 chunks)
   - Citation formatting (< 0.01s for 100 citations)

6. **Caching & Tokens**
   - Cache hit/miss behavior
   - Deterministic cache keys
   - Token tracking
   - Batch processing
   - Cache-friendly prompt design
   - Token budget compliance

## Performance Targets

| Component | Target | Notes |
|-----------|--------|-------|
| Chunking | < 1s / 100 pages | Fixed or section-aware |
| Normalization | > 10 MB/s | Text processing throughput |
| Retrieval | < 0.1s / 1000 chunks | Filtering and sorting |
| Citation Format | < 0.01s / 100 cites | Reference formatting |
| Memory | < 50MB | Peak usage for typical workload |
| Cache Hit Rate | > 80% | For repeated benchmark runs |
| Token Budget | < 4K / query | Input + output tokens |

## CI/CD Integration

```bash
# Quick smoke test (for CI)
python scripts/run_tests.py --fast --quiet

# Full test suite (for releases)
python scripts/run_tests.py
```

## Troubleshooting

### Import Errors
```bash
# Install dependencies
pip install -r requirements.txt

# Verify imports
python scripts/test_structure.py
```

### API Tests Failing
```bash
# Set API key
export DEEPSEEK_API_KEY="your-key-here"

# Or skip API tests
python scripts/run_tests.py --fast
```

### Data Directory Issues
```bash
# Create required directories
mkdir -p data/cache data/indices data/parsed data/pdf
```

## Adding New Tests

1. Create test file in `tests/` directory
2. Follow naming convention: `test_*.py`
3. Use unittest.TestCase as base class
4. Add module to appropriate list in `scripts/run_tests.py`
5. Document performance targets if applicable

## Test Philosophy

- **Fast feedback**: Unit tests should be fast (< 0.1s each)
- **Realistic data**: Use realistic domain content (derivatives, Greeks, etc.)
- **Clear assertions**: Every test should have clear pass/fail criteria
- **Isolated**: Tests should not depend on external state
- **Documented**: Complex tests should explain what they verify

## Performance Benchmarking

For detailed performance profiling:

```bash
# Run with profiling
python -m cProfile -o profile.stats scripts/run_tests.py --performance

# Analyze results
python -m pstats profile.stats
```

## Coverage Report

To generate coverage report:

```bash
# Install coverage tool
pip install coverage

# Run with coverage
coverage run -m pytest tests/

# Generate report
coverage report
coverage html
```

