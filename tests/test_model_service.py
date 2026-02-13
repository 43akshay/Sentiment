import os
import pytest
from model_service import ModelService

def test_model_not_found():
    service = ModelService(model_path="./non_existent_model")
    assert service.model is None
    result = service.predict("hello")
    assert "error" in result
    assert "Model not loaded" in result["error"]
