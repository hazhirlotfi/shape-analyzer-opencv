import cv2, numpy as np
from matplotlib import pyplot as plt

!wget -O image_link.jpg https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQQY1KyLbiY-crkH7oVXKnZ9yVqU1RCyzxA_F8Hr89L3XSVjIdtSQFkphbp&s=10 # put a desired link.

class ShapeAnalyzer:
  def __init__(self):
    self.image, self.canny_image, self.contours = None, None, None
    self.hierarchy = None

  def load_image(self):
    try:
      self.image = cv2.imread('image_link.jpg')
    except FileNotFoundError:
      print("File Not Found.\n")

  def preprocess(self):
    gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    self.canny_image = cv2.Canny(gray_image, 70, 100)

  def detect_contours(self):
    self.contours, self.hierarchy = cv2.findContours(self.canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

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

