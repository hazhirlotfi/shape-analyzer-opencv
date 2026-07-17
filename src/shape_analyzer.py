import cv2, numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


class ShapeAnalyzer:
    def __init__(self, image_path):
        self.image_path = Path(image_path)
        if self.image_path.exists() is False:
            raise FileNotFoundError(f"File couldn't be found: {self.image_path}")
        self._reset_state()

    def _reset_state(self):
        self.image = None
        self.canny_image = None
        self.contours = None
        self.hierarchy = None
        self.contour_count = None

    def load_image(self):
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise FileNotFoundError(f"File couldn't be found: {self.image_path}")

    def preprocess(self, low=70, high=100):
        if self.image is None:
            raise RuntimeError("preprocess() called before load_image()")
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        anti_noise_removal = cv2.bilateralFilter(gray_image, 5, 60, 60)
        self.canny_image = cv2.Canny(anti_noise_removal, low, high)

    def detect_contours(self):
        if self.canny_image is None:
            raise RuntimeError("detect_contours() called before preprocess()")
        self.contours, self.hierarchy = cv2.findContours(
            self.canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        self.contour_count = len(self.contours)
        print(f"{self.image_path.name} has {self.contour_count} contours.")

    def draw_results(self):
        if self.contours is None:
            raise RuntimeError("draw_results() called before detect_contours()")
        cv2.drawContours(self.image, self.contours, -1, (255, 0, 0), 5)

    def show_results(self):
        size = 10
        h, w = self.image.shape[:2]
        aspect_ratio = w / h
        plt.figure(figsize=(size * aspect_ratio, size))
        plt.title("CONTOURS DRAWN")
        plt.imshow(self.image)
        plt.show()

    def save_result(self):
        if self.image is None:
            raise RuntimeError("save_result() called before load_image()")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(RESULTS_DIR / "result.jpg"), self.image)

    def process(self):
        self.load_image()
        self.preprocess()
        self.detect_contours()

    def visualize(self):
        self.draw_results()
        self.show_results()

    def run(self, image_path=None):
        if image_path is not None:
            self.image_path = Path(image_path)

        self.process()
        self.visualize()
        self.save_result()
