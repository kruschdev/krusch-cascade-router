#!/usr/bin/env python3
"""
Cross-Language Parity Test: Validates that benchmark/krusch_cascade_adapter.py
produces 100% identical routing decisions to the TypeScript classifier
on test/fixtures/routing-spec.json.
"""

import json
import os
import sys

# Add RouterArena router directory to path so BaseRouter can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
router_arena_dir = os.path.join(current_dir, "RouterArena")
sys.path.insert(0, router_arena_dir)
sys.path.insert(0, current_dir)

try:
    from krusch_cascade_adapter import KruschCascadeRouter
except ImportError:
    # Fallback to mock BaseRouter if RouterArena submodule is not initialized
    class MockBaseRouter:
        def __init__(self, name):
            self.config = {}
    
    import types
    mod = types.ModuleType("router_inference.router.base_router")
    mod.BaseRouter = MockBaseRouter
    sys.modules["router_inference.router.base_router"] = mod
    from krusch_cascade_adapter import KruschCascadeRouter

def run_parity_tests():
    fixtures_path = os.path.join(current_dir, "..", "test", "fixtures", "routing-spec.json")
    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    router = KruschCascadeRouter()
    passed = 0
    failures = []

    for item in fixtures:
        query = item["query"]
        expected_model = item["expected_model"]
        predicted = router._get_prediction(query)

        if predicted == expected_model:
            passed += 1
        else:
            failures.append({
                "query": query,
                "expected": expected_model,
                "got": predicted
            })

    print(f"========================================")
    print(f"🎯 Parity Test Results (Python Adapter)")
    print(f"========================================")
    print(f"Total Fixtures: {len(fixtures)}")
    print(f"Passed:         {passed}/{len(fixtures)} (100% required)")
    print(f"========================================")

    if failures:
        print("\n❌ Failures detected:")
        for fail in failures:
            print(f"  - Query:    {fail['query']}")
            print(f"    Expected: {fail['expected']}")
            print(f"    Got:      {fail['got']}\n")
        sys.exit(1)
    else:
        print("✅ 100% Parity Verified between Python Adapter and Shared Fixtures!\n")

if __name__ == "__main__":
    run_parity_tests()
