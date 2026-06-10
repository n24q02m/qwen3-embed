import sys
from unittest.mock import patch

def test():
    try:
        import llama_cpp
        print("Imported")
    except ImportError:
        print("ImportError")

sys.modules["llama_cpp"] = object()
print("Initial state:")
test()

print("With None in sys.modules:")
with patch.dict(sys.modules, {"llama_cpp": None}):
    test()

print("After patch:")
test()
