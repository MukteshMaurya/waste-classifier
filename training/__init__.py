"""Training and model-export pipeline for the AI Waste Classifier.

This package is intentionally separate from the deployed backend: the API only
ever loads the ONNX artefact produced by ``training/export_model.py``.
"""
