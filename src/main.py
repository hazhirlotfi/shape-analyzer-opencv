from shape_analyzer import ShapeAnalyzer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
image_path = PROJECT_ROOT / "images"

for img_path in image_path.iterdir():
    shape = ShapeAnalyzer(img_path)
    shape.run()
 