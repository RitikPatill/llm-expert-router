import importlib


def test_package_importable():
    mod = importlib.import_module("llm_expert_router")
    assert mod.__version__ == "0.1.0"
