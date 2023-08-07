class ImagePreprocessor:

    def __init__(self, image):
        self.image = image

    def get_rotation_angle(self):
        """
        Detects the average rotation angle of the image using HoughLines
        while considering lines that deviate around 30 degrees from the vertical.
        """
        # Convert the image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if not lines.any():
            return 0

        angles = []
        for line in lines:
            rho, theta = line[0]
            
            # Convert the angle from radians to degrees
            angle = theta * 180.0 / np.pi - 90  # Convert from [0, 180] to [-90, 90]
            
            # Filter to keep angles within a rotation error of 30 degrees from vertical
            if -60 < angle < 60:
                angles.append(angle)

        # Compute the average angle
        if not angles:
            return 0

        avg_angle = sum(angles) / len(angles)
        return avg_angle

    def rotate_image(self, angle):
        """
        Rotates the image by the specified angle.
        """
        height, width = self.image.shape[:2]
        center = (width // 2, height // 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1)
        
        # Perform the rotation
        rotated = cv2.warpAffine(self.image, M, (width, height))
        
        return rotated

    def fix_rotation(self):
        """
        Fixes the rotation of the image.
        """
        angle = self.get_rotation_angle()
        return self.rotate_image(-angle)  # We negate the angle to correct the rotation

    def draw_lines(self):
        """
        Draws the Hough Lines on the image and returns the resultant image.
        """
        # Convert the image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if not lines.any():
            return self.image.copy()

        line_image = self.image.copy()
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return line_image

