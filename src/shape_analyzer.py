import cv2, numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_ROOT / "images" / "test1.jpg"


class ShapeAnalyzer:
    def __init__(self):
        self.image = None
        self.canny_image = None
        self.contours = None
        self.hierarchy = None

    def load_image(self):
        self.image = cv2.imread(str(IMAGE_PATH))
        if self.image is None:
            raise FileNotFoundError(f"File couldn't be found.\n")

    def preprocess(self, low = 70, high = 100):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.canny_image = cv2.Canny(gray_image, low, high)

    def detect_contours(self):
        self.contours, self.hierarchy = cv2.findContours(
            self.canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

    def draw_results(self):
        size = 10
        h, w = self.image.shape[:2]
        aspect_ratio = w / h
        plt.figure(figsize=(size * aspect_ratio, size))
        plt.title("result")
        cv2.drawContours(self.image, self.contours, -1, (255, 0, 0), 5)
        plt.imshow(self.image)
        plt.show()

    def run(self):
        self.load_image()
        self.preprocess()
        self.detect_contours()
        self.draw_results()
