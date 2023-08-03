import cv2
import numpy as np

class AnalyzedImage:
    def __init__(self, file_path):
        self.image = cv2.imread(file_path)
        self.original_image = self.image.copy() # Keep a copy of the original

    def change_contrast(self, alpha, beta=0):
        """Adjust the contrast of the image.
        alpha -- Contrast control (1.0-3.0)
        beta  -- Brightness control (0-100)
        """
        self.image = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=beta)

    def crop(self, x, y, width, height):
        """Crop the image to the specified coordinates."""
        self.image = self.image[y:y+height, x:x+width]

    def resize(self, width, height):
        self.image = cv2.resize(self.image, (width, height))

    def to_grayscale(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def blur(self, kernel_size=(5, 5)):
        self.image = cv2.GaussianBlur(self.image, kernel_size, 0)

    def threshold(self, threshold_value=0, max_value=255, threshold_type=cv2.THRESH_BINARY):
        _, self.image = cv2.threshold(self.image, threshold_value, max_value, threshold_type)

    def invert(self):
        self.image = cv2.bitwise_not(self.image)

    def detect_edges(self):
        self.image = cv2.Canny(self.image, 100, 200)

    def dilate(self, kernel_size=(5, 5)):
        kernel = np.ones(kernel_size, np.uint8)
        self.image = cv2.dilate(self.image, kernel, iterations=1)

    def erode(self, kernel_size=(5, 5)):
        kernel = np.ones(kernel_size, np.uint8)
        self.image = cv2.erode(self.image, kernel, iterations=1)

    def find_contours(self):
        _, contours, _ = cv2.findContours(self.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def draw_contours(self, contours, color=(0, 255, 0), thickness=2):
        cv2.drawContours(self.image, contours, -1, color, thickness)

    def find_corners(self, contours):
        # Find the corners of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.1 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        return approx

    def draw_corners(self, corners, color=(0, 255, 0), thickness=2):
        cv2.drawContours(self.image, [corners], -1, color, thickness)

    def find_bounding_box(self, corners):
        # Find the bounding box of the corners
        x, y, width, height = cv2.boundingRect(corners)
        return x, y, width, height

    def draw_bounding_box(self, x, y, width, height, color=(0, 255, 0), thickness=2):
        cv2.rectangle(self.image, (x, y), (x+width, y+height), color, thickness)

    

    def normalize_intensity(self):
        """Normalize the image intensity values to stretch them over the range [0, 255]."""
        min_val = np.min(self.image)
        max_val = np.max(self.image)
        self.image = ((self.image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    def equalize_histogram(self):
        if len(self.image.shape) == 2:  # Grayscale image
            self.image = cv2.equalizeHist(self.image)
        else:  # Color image
            channels = cv2.split(self.image)
            for i in range(len(channels)):
                channels[i] = cv2.equalizeHist(channels[i])
            self.image = cv2.merge(channels)


    def preprocess_for_recognition(self):
        """Preprocess the image for number recognition.
        This might include resizing, thresholding, etc.
        """
        # Convert to grayscale
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Thresholding
        _, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Additional preprocessing steps can be added here
        self.image = thresholded

    def get_image(self):
        """Return the current image."""
        return self.image

    def reset_image(self):
        """Reset the image to the original."""
        self.image = self.original_image.copy()


