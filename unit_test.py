#!/usr/bin/env python3
"""
unit_test.py

Run the repository unit test suite from the project root.
This file is intended to be executed before other code to verify the current state.
"""
import os
import sys
import unittest


def main() -> int:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=repo_root, pattern="test.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
