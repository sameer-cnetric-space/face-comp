import cv2
import numpy as np
from typing import Tuple  # Importing Tuple from typing module
from liveCheckModules.liveness_utils import (
    LivenessCheckResult,
)  # Import from liveness_utils


def check_sharpness(face_image: np.ndarray) -> LivenessCheckResult:
    """Check the sharpness of the face image"""
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    passed = score > 50.0  # Adjust the threshold as needed
    return passed, score, {}


def check_face_symmetry(
    image: np.ndarray, face_location: Tuple[int, int, int, int]
) -> LivenessCheckResult:
    """Check face symmetry"""
    # Add actual implementation logic here
    score = 0.5  # Example score
    passed = score > 0.35
    return passed, score, {}


def check_skin_texture(face_image: np.ndarray) -> LivenessCheckResult:
    """Check the skin texture of the face image"""
    # Implement actual logic
    score = 0.6  # Example score
    passed = score > 0.4
    return passed, score, {}


def check_moire_patterns(face_image: np.ndarray) -> LivenessCheckResult:
    """Check for moire patterns in the face image"""
    # Implement actual logic
    score = 0.1  # Example score
    passed = score < 0.2
    return passed, score, {}


def check_depth_variation(face_image: np.ndarray) -> LivenessCheckResult:
    """Check depth variation in the face image"""
    # Implement actual logic
    score = 0.3  # Example score
    passed = score > 0.25
    return passed, score, {}


def check_noise_patterns(
    face_image: np.ndarray, sharpness_score: float, logger
) -> LivenessCheckResult:
    """Check noise patterns using the variance of the Laplacian."""
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # Calculate the variance of the Laplacian (high values indicate more noise or sharpness)
    noise_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Log the noise score
    logger.info(f"Calculated Noise (Laplacian Variance): {noise_score}")

    # Dynamically adjust thresholds based on sharpness score
    if sharpness_score > 1000:
        noise_threshold_low = 200.0  # Allow higher noise for sharper images
    elif sharpness_score > 500:
        noise_threshold_low = 100.0  # Medium threshold for noise
    else:
        noise_threshold_low = 50.0  # Be strict on noise for less sharp images

    # Log the noise threshold
    logger.info(f"Noise Threshold Low: {noise_threshold_low}")

    passed = noise_score > noise_threshold_low
    return LivenessCheckResult(passed, noise_score, {}, 0.0)


def check_eye_blink(
    image: np.ndarray, face_location: Tuple[int, int, int, int]
) -> LivenessCheckResult:
    """Check for eye blink in the image"""
    # Implement actual logic for eye blink detection
    score = 0.4  # Example score
    passed = score > 0.3
    return passed, score, {}


def check_face_size(face_image: np.ndarray) -> LivenessCheckResult:
    """Check the size of the detected face"""
    face_size_ratio = (
        face_image.shape[0] * face_image.shape[1] / (1024 * 768)
    )  # Example logic
    passed = face_size_ratio > 0.1
    return passed, face_size_ratio, {}


def compute_snr(image: np.ndarray) -> float:
    """Compute the signal-to-noise ratio for the image, with logging."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_signal = np.mean(gray)  # Mean of the signal
    noise_std = np.std(gray)  # Standard deviation (assumed to represent noise)

    if noise_std == 0:
        return 0.0  # Logically return 0 SNR if there's no variance

    snr = mean_signal / noise_std
    return snr


def edge_based_noise_detection(face_image: np.ndarray, logger) -> LivenessCheckResult:
    """Estimate noise based on the amount of edges detected in the image."""
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    edge_ratio = np.sum(edges) / (
        face_image.shape[0] * face_image.shape[1]
    )  # Ratio of edge pixels to total pixels

    # Log the edge ratio
    # logger.info(f"Edge Ratio: {edge_ratio}")

    # Adjust the threshold based on your observations
    # e.g., anything below 5 is non-live, and above 5 is live
    passed = edge_ratio > 5.0

    return LivenessCheckResult(passed, edge_ratio, {}, 0.0)
