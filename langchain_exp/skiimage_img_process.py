import cv2
import numpy as np
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, median
from skimage.morphology import disk, skeletonize
from os.path import isfile
def load_image(path):
    return rgb2gray(io.imread(path, as_gray=True))

# Binarization

def binarize(image):
    thresh = threshold_otsu(image)
    binary = image <= thresh
    return binary

# Skew correction

def correct_skew(image):
    # Assuming that the text is horizontal, we can find the most dominant angle
    # This is a simplified version and may require adjustment
    edges = cv2.Canny(image.astype(np.uint8)*255, 30, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    angles = []
    if lines is not None:
        for [[x1, y1, x2, y2]] in lines:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
    median_angle = np.median(angles)
    return rotate(image, median_angle)

# Noise removal

def remove_noise(image):
    return median(image, disk(1))

# Thinning / Skeletonization

def skeletonize_image(image):
    return skeletonize(image)

# Combine all preprocessing steps

def preprocess_image(path):
    image = load_image(path)
    binary = binarize(image)
    corrected = correct_skew(binary)
    noise_free = remove_noise(corrected)
    thinned = skeletonize_image(noise_free)
    return thinned

# Example usage
path = 'PLACEHOLDER_PATH'  # replace with your image path
if isfile(path):
    processed_image = preprocess_image(path)
    # Save the processed image
    cv2.imwrite('processed_' + path, processed_image * 255)
    print('Processed image saved.')
elif path == 'PLACEHOLDER_PATH':
    print('Please replace the placeholder path with a valid image path.')
else:
    print(f'Image not found at path: {path}')
Please replace the placeholder path with a valid image path.
