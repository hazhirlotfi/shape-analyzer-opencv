from shape_analyzer import ShapeAnalyzer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
image_path = PROJECT_ROOT / "images" / "test1.jpg"

shape = ShapeAnalyzer(image_path)
shape.run()
