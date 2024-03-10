# Overview
This script is designed to enrich an existing ONNX model by inserting a Reshape operation to adapt the input tensor shape.

### Constants

- `MODEL_PATH`: Path to the input ONNX model.
- `UNET_INPUT_SHAPE`: Shape information required for the U-Net model.
- `OUTPUT_MODEL_PATH`: Path to save the enriched ONNX model.

### Dependencies
- `onnx`: The Open Neural Network Exchange library for working with ONNX models.



