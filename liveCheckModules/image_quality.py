# ./liveCheckModules/image_quality.py
import cv2
import numpy as np
import face_recognition


def is_image_quality_acceptable(
    image: np.ndarray,
    min_resolution: int = 100,
    min_face_size: int = 40,
    min_sharpness: float = 50.0,
) -> tuple[bool, str]:
    """
    Checks if the image meets quality standards based on resolution, face detectability, and sharpness.
    """
    # Check resolution
    if image.shape[0] < min_resolution or image.shape[1] < min_resolution:
        return False, "Image resolution too low."

    # Detect faces
    face_locations = face_recognition.face_locations(image, model="hog")
    if not face_locations:
        return False, "No faces detected."

    # Ensure each detected face meets minimum size and clarity requirements
    for top, right, bottom, left in face_locations:
        face_width, face_height = right - left, bottom - top
        if face_width >= min_face_size and face_height >= min_face_size:
            face_region = image[top:bottom, left:right]

            # Check sharpness (e.g., Laplacian variance)
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            sharpness_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            if sharpness_score < min_sharpness:
                return (
                    False,
                    f"Face detected but too low in quality (sharpness: {sharpness_score:.2f}).",
                )

            return True, "Image quality is acceptable."

    return False, "Face(s) too small or unclear."
