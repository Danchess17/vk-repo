import cv2
import numpy as np
import os

def detect_logo(image_path, logo_paths, threshold=0.7):
    """
    Detects if the image contains a logo from the provided logo samples.

    Args:
        image_path (str): Path to the image to analyze.
        logo_paths (list): List of paths to the logo images.
        threshold (float): Threshold for the matching score.

    Returns:
        bool: True if a logo is detected, False otherwise.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    kp_img, des_img = sift.detectAndCompute(img, None)

    for logo_path in logo_paths:
        logo = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
        kp_logo, des_logo = sift.detectAndCompute(logo, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_img, des_logo, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)

        if len(good_matches) > 20:  # Adjust the number of matches
            return True

    return False


# Main
image_dir = "images"
logo_dir = "logos"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
logo_paths = [os.path.join(logo_dir, f) for f in os.listdir(logo_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Logo paths: {logo_paths}")
print(f"Image paths: {image_paths}")

for image_path in image_paths:
    if detect_logo(image_path, logo_paths):
        print(f"Logo found in: {image_path}")
    else:
        print(f"Logo NOT found in: {image_path}")
