import os
import pytest

# This runs before pytest even looks at your test files
os.environ["GOOGLE_API_KEY"] = "dummy-test-key"
os.environ["GEMINI_API_KEY"] = "dummy-test-key"