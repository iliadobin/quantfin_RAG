#!/usr/bin/env python3
"""
Unified test runner for QA Assistant.

Runs all test suites and generates a comprehensive report.

Usage:
    python scripts/run_tests.py                    # Run all tests
    python scripts/run_tests.py --unit             # Run only unit tests
    python scripts/run_tests.py --integration      # Run only integration tests
    python scripts/run_tests.py --performance      # Run only performance tests
    python scripts/run_tests.py --cache            # Run only cache/token tests
    python scripts/run_tests.py --fast             # Skip slow tests (no API calls)
"""
import sys
import argparse
import unittest
from pathlib import Path
import time
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env (so DEEPSEEK_API_KEY is visible to tests without manual export)
try:
    from dotenv import load_dotenv

    load_dotenv(project_root / ".env", override=False)
except Exception:
    # Keep test runner usable even if python-dotenv isn't installed.
    pass


class TestSuites:
    """Test suite definitions."""
    
    UNIT_TESTS = [
        "tests.test_models",
        "tests.test_contracts",
        "tests.test_ingest_pipeline",
    ]
    
    INTEGRATION_TESTS = [
        "tests.test_integration",
    ]
    
    PERFORMANCE_TESTS = [
        "tests.test_performance",
    ]
    
    CACHE_TESTS = [
        "tests.test_cache_tokens",
    ]
    
    @classmethod
    def get_all_tests(cls):
        """Get all test modules."""
        return cls.UNIT_TESTS + cls.INTEGRATION_TESTS + cls.PERFORMANCE_TESTS + cls.CACHE_TESTS


def load_test_suite(module_names):
    """Load test suite from module names."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=[''])
            tests = loader.loadTestsFromModule(module)
            suite.addTests(tests)
        except ImportError as e:
            print(f"Warning: Could not load {module_name}: {e}")
    
    return suite


def run_test_suite(suite, verbosity=2):
    """Run test suite and return results."""
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def print_summary(results, elapsed_time):
    """Print test summary."""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {results.testsRun}")
    print(f"Successes: {results.testsRun - len(results.failures) - len(results.errors)}")
    print(f"Failures: {len(results.failures)}")
    print(f"Errors: {len(results.errors)}")
    print(f"Skipped: {len(results.skipped)}")
    print(f"Time: {elapsed_time:.2f}s")
    print("="*80)
    
    if results.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        
        if results.failures:
            print("\nFailed tests:")
            for test, traceback in results.failures:
                print(f"  - {test}")
        
        if results.errors:
            print("\nTests with errors:")
            for test, traceback in results.errors:
                print(f"  - {test}")
    
    print("="*80 + "\n")


def check_environment():
    """Check environment and dependencies."""
    print("Checking environment...")
    
    issues = []
    
    # Check if data directories exist
    data_dir = project_root / "data"
    if not data_dir.exists():
        issues.append("data/ directory not found - some tests may fail")
    
    # Check for API key (for cache tests)
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("  ⚠️  DEEPSEEK_API_KEY not set - cache/API tests will be skipped")
    else:
        print("  ✓ DEEPSEEK_API_KEY found")
    
    # Check critical imports
    try:
        import yaml
        print("  ✓ yaml")
    except ImportError:
        issues.append("pyyaml not installed")
    
    try:
        import pydantic
        print("  ✓ pydantic")
    except ImportError:
        issues.append("pydantic not installed")
    
    if issues:
        print("\n⚠️  Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
        print()
    else:
        print("  ✓ All dependencies found\n")
    
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Run QA Assistant tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--cache", action="store_true", help="Run only cache/token tests")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests (no API calls)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    # Determine verbosity
    verbosity = 2
    if args.verbose:
        verbosity = 2
    elif args.quiet:
        verbosity = 0
    
    # Set environment variable for fast mode
    if args.fast:
        os.environ["SKIP_API_TESTS"] = "1"
        print("⚡ Fast mode: Skipping API-dependent tests\n")
    
    # Check environment
    env_ok = check_environment()
    
    # Determine which test suites to run
    test_modules = []
    
    if args.unit:
        test_modules.extend(TestSuites.UNIT_TESTS)
        print("Running UNIT tests...\n")
    elif args.integration:
        test_modules.extend(TestSuites.INTEGRATION_TESTS)
        print("Running INTEGRATION tests...\n")
    elif args.performance:
        test_modules.extend(TestSuites.PERFORMANCE_TESTS)
        print("Running PERFORMANCE tests...\n")
    elif args.cache:
        test_modules.extend(TestSuites.CACHE_TESTS)
        print("Running CACHE/TOKEN tests...\n")
    else:
        # Run all tests
        test_modules = TestSuites.get_all_tests()
        print("Running ALL tests...\n")
    
    # Load and run tests
    start_time = time.time()
    suite = load_test_suite(test_modules)
    results = run_test_suite(suite, verbosity=verbosity)
    elapsed = time.time() - start_time
    
    # Print summary
    print_summary(results, elapsed)
    
    # Exit with appropriate code
    sys.exit(0 if results.wasSuccessful() else 1)


if __name__ == "__main__":
    main()

