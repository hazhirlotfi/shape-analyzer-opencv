import cv2, numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import json

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
        self.gray_image = None
        self.canny_image = None
        self.contours = None
        self.hierarchy = None
        self.contour_count = None
        self.filtered_area = []
        self.shape_data = []

    def load_image(self):
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise FileNotFoundError(f"File couldn't be found: {self.image_path}")

    def preprocess(self, low=20, high=90, close_size=5, close_iterations=2):
        if self.image is None:
            raise RuntimeError("preprocess() called before load_image()")
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        gaussian_blur_filter = cv2.GaussianBlur(self.gray_image, (5, 5), 0)

        median_blur_filter = cv2.medianBlur(gaussian_blur_filter, (7))

        edges = cv2.Canny(median_blur_filter, low, high)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (close_size, close_size))

        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=2)

        self.canny_image = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=close_iterations,
        )

    def detect_contours(self):
        if self.canny_image is None:
            raise RuntimeError("detect_contours() called before preprocess()")
        self.contours, self.hierarchy = cv2.findContours(
            self.canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        self.contour_count = len(self.contours)
        print(f"{self.image_path.name} has {self.contour_count} contours.")

        for contour in self.contours:
            perimeter = cv2.arcLength(contour, True)
            if self.shape_data and self.shape_data[0]["name"] == "circle / oval":
                epsilon = 0.0001 * perimeter
            else:
                epsilon = 0.009 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            points = len(approx)

            if points == 3:
                shape_name = "triangle"
            elif points == 4:
                shape_name = "quadrilateral (probably square :)"
            elif points == 5:
                shape_name = "pentagon"
            else:
                shape_name = "circle / oval"

            self.shape_data.append(
                {
                    "contour": contour,
                    "approx": approx,
                    "name": shape_name,
                    "points": points,
                    "area": cv2.contourArea(contour),
                    "perimeter": perimeter,
                }
            )

    def filter_contours(
        self,
        relative_area=0.03,
        relative_perimeter=0.1,
        min_area=None,
        min_perimeter=None,
    ):
        areas = [shape["area"] for shape in self.shape_data]
        perimeters = [shape["perimeter"] for shape in self.shape_data]

        if relative_area and min_area is None and areas:
            min_area = relative_area * max(areas)
        if relative_perimeter and min_perimeter is None and perimeters:
            min_perimeter = relative_perimeter * max(perimeters)

        self.filtered_area = []
        for each_shape in self.shape_data:
            passes_area = min_area is not None and each_shape["area"] > min_area
            passes_perimeter = (
                min_perimeter is not None and each_shape["perimeter"] > min_perimeter
            )
            if passes_area or passes_perimeter:
                self.filtered_area.append(each_shape)

    def draw_results(self):
        if self.contours is None:
            raise RuntimeError("draw_results() called before detect_contours()")

        for i, shapes in enumerate(self.filtered_area):
            cv2.drawContours(self.image, [shapes["approx"]], -1, (255, 0, 0), 5)

    def show_results(self):
        size = 10
        h, w = self.image.shape[:2]
        aspect_ratio = w / h
        plt.figure(figsize=(size * aspect_ratio, size))
        plt.title("CONTOURS DRAWN")
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
        plt.show()

    def save_result(self):
        if self.image is None:
            raise RuntimeError("save_result() called before load_image()")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(RESULTS_DIR / self.image_path.name), self.image)

    def json_result(self):
        json_data = []
        for each_shape in self.shape_data:
            json_pass = {
                "name": each_shape["name"],
                "array": each_shape["approx"].tolist(),
                "points": int(each_shape["points"]),
                "area": float(each_shape["area"]),
                "perimeter": float(each_shape["perimeter"]),
            }
            json_data.append(json_pass)
        json_logs = json.dumps(json_data)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = RESULTS_DIR / f"{self.image_path.stem}_results.json"

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"JSON saved to: {json_path}")

    def process(self):
        self.load_image()
        self.preprocess()
        self.detect_contours()
        self.filter_contours()
        self.json_result()

    def visualize(self):
        self.draw_results()
        self.show_results()
        self.save_result()

    def run(self, image_path=None):
        if image_path is not None:
            self.image_path = Path(image_path)

        self.process()
        self.visualize()
