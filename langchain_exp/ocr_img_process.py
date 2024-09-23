import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import matplotlib.pyplot as plt
class OCRPreprocessor:
    def __init__(self, img_path):
        self.image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.binary_image = None
        self.corrected_image = None
        self.noise_removed_image = None
        self.thinned_image = None

    def binarize(self):
        _, self.binary_image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self.binary_image

    def correct_skew(self):
        def find_score(arr, angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            hist = np.sum(data, axis=1)
            score = np.sum((hist[1:] - hist[:-1]) ** 2)
            return hist, score

        delta = 1
        limit = 5
        angles = np.arange(-limit, limit+delta, delta)
        scores = []
        for angle in angles:
            hist, score = find_score(self.binary_image, angle)
            scores.append(score)

        best_score = max(scores)
        best_angle = angles[scores.index(best_score)]
        self.corrected_image = inter.rotate(self.binary_image, best_angle, reshape=False, order=0)
        return self.corrected_image

    def remove_noise(self):
        kernel = np.ones((1, 1), np.uint8)
        self.noise_removed_image = cv2.morphologyEx(self.corrected_image, cv2.MORPH_OPEN, kernel)
        return self.noise_removed_image

    def thin(self):
        size = np.size(self.noise_removed_image)
        skel = np.zeros(self.noise_removed_image.shape, np.uint8)

        ret, img = cv2.threshold(self.noise_removed_image, 127, 255, 0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while (not done):
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        self.thinned_image = skel
        return self.thinned_image

    def save_image(self, filename):
        cv2.imwrite(filename, self.thinned_image)

if __name__ == '__main__':
    img_path = 'path/to/your/image.jpg'
    preprocessor = OCRPreprocessor(img_path)
    preprocessor.binarize()
    preprocessor.correct_skew()
    preprocessor.remove_noise()
    preprocessor.thin()
    preprocessor.save_image('processed_image.jpg')
    print('Image processing complete and saved.')
