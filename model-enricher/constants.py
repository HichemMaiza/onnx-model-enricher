from pathlib import Path

UNET_INPUT_SHAPE = [3, 320, 320]
MODEL_PATH = Path("../resources/u2net.onnx").absolute()
OUTPUT_MODEL_PATH = Path("../resources/u2net_enriched.onnx").absolute()