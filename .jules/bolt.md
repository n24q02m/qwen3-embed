## 2025-05-14 - [TEST] Test coverage for export_to_onnx error path
**Learning:** Mocking sys.modules is an effective way to test error paths for optional dependencies that might be present in the environment.
**Action:** Added a test case in tests/test_export.py that mocks optimum.exporters.onnx to verify the ImportError is raised correctly.
