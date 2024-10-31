from pathlib import Path
import logging
import cv2
import numpy as np
import face_recognition
from typing import Tuple, List, Optional, Dict, Union
from concurrent.futures import ThreadPoolExecutor
import time
from liveCheckModules.liveness_utils import (
    LivenessCheckResult,
)  # Import from liveness_utils
from liveCheckModules.liveness_checks import (
    check_sharpness,
    check_face_symmetry,
    check_skin_texture,
    check_moire_patterns,
    check_depth_variation,
    edge_based_noise_detection,
    check_eye_blink,
    check_face_size,
)


class LivenessDetector:
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger("LivenessDetector")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Set the logging level based on the config (debug or not)
        if config and config.get("debug", False):
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        """
        Initialize LivenessDetector with configurable parameters.
        """
        self.config = {
            "thresholds": {
                "sharpness": 50.0,
                "symmetry": 0.35,
                "texture": 0.4,
                "moire": 0.2,
                "depth": 0.25,
                "noise": (2.0, 30.0),
                "blink": 0.3,
                "face_size": 0.1,
            },
            "debug": False,
            "debug_dir": "debug_output",
            "max_workers": 4,
            "face_detection_model": "hog",
            "min_face_size": 30,
            "cache_results": True,
            "resize_for_detection": True,
        }

        if config:
            self.config.update(config)

        self._cache = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config["max_workers"])

    def __enter__(self):
        """Context manager entry: just return self"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit: shutdown the executor and clear cache"""
        self.executor.shutdown(wait=True)  # Shutdown the thread pool executor safely
        self.clear_cache()

    def clear_cache(self):
        """Clear the results cache"""
        self._cache.clear()

    def process_image(
        self, image_input: Union[str, np.ndarray, Path]
    ) -> Tuple[bool, str, Dict]:
        start_time = time.time()

        try:
            image = self._load_and_preprocess_image(image_input)
            if image is None:
                return False, "Failed to load or preprocess image", {}

            results = {
                "checks": {},
                "debug_info": {"image_size": image.shape, "processing_time": {}},
            }

            face_locations = self._detect_faces_robust(image)
            if not face_locations:
                return False, "No valid face detected", results

            face_location = face_locations[0]
            face_image = self._extract_face_region(image, face_location)

            check_results = self._run_parallel_checks(image, face_image, face_location)

            is_live = True
            failed_checks = []
            for check_name, result in check_results.items():
                results["checks"][check_name] = {
                    "passed": bool(result.passed),
                    "score": float(result.score),
                    "details": result.details,
                    "duration": result.duration,
                }
                if not result.passed:
                    is_live = False  # If any check fails, mark liveness as failed
                    failed_checks.append(check_name)

            # Prepare the message
            message = (
                "All liveness checks passed"
                if is_live
                else f"Failed checks: {', '.join(failed_checks)}"
            )
            results["debug_info"]["total_processing_time"] = time.time() - start_time

            return is_live, message, results

        except Exception as e:
            return False, f"Processing error: {str(e)}", {}

    def _load_and_preprocess_image(
        self, image_input: Union[str, np.ndarray, Path]
    ) -> Optional[np.ndarray]:
        """Load and preprocess the input image."""
        try:
            # self.logger.info(f"Attempting to load image: {image_input}")

            if isinstance(image_input, np.ndarray):
                # self.logger.info("Image input is already a numpy array.")
                image = image_input
            else:
                # Convert to string in case Path object is passed
                image_input = str(image_input)
                self.logger.info(f"Reading image from path: {image_input}")
                image = cv2.imread(image_input)

            # Check if the image was loaded correctly
            if image is None or image.size == 0:
                self.logger.error(f"Failed to load image from path: {image_input}")
                return None

            # Resize image for faster processing, if needed (configurable)
            if self.config["resize_for_detection"]:
                max_dim = (
                    1024  # Example: Resize if the largest dimension exceeds 1024 pixels
                )
                if max(image.shape) > max_dim:
                    scale_factor = max_dim / max(image.shape)
                    new_size = (
                        int(image.shape[1] * scale_factor),
                        int(image.shape[0] * scale_factor),
                    )
                    image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
                    # self.logger.info(f"Resized image to: {new_size}")

            return image

        except Exception as e:
            self.logger.error(
                f"Error in loading or preprocessing image: {e}", exc_info=True
            )
            return None

    def _detect_faces_robust(self, image: np.ndarray) -> List:
        """Detect faces robustly using the face_recognition library."""
        try:
            # self.logger.info("Starting face detection.")

            # Detect faces using the configured model (hog or cnn)
            model = self.config["face_detection_model"]
            face_locations = face_recognition.face_locations(image, model=model)

            if not face_locations:
                self.logger.info("No faces detected.")
                return []

            # Check face size and filter small faces
            valid_faces = []
            for loc in face_locations:
                top, right, bottom, left = loc
                face_height = bottom - top
                face_width = right - left
                if (
                    face_height >= self.config["min_face_size"]
                    and face_width >= self.config["min_face_size"]
                ):
                    valid_faces.append(loc)

            # Sort faces by size (largest face first)
            valid_faces.sort(key=lambda x: (x[2] - x[0]) * (x[1] - x[3]), reverse=True)

            # self.logger.info(f"Detected {len(valid_faces)} valid faces.")
            return valid_faces

        except Exception as e:
            self.logger.error(f"Error in face detection: {e}", exc_info=True)
            return []

    def _extract_face_region(
        self, image: np.ndarray, face_location: Tuple[int, int, int, int]
    ) -> np.ndarray:
        top, right, bottom, left = face_location
        return image[top:bottom, left:right]

    def _run_parallel_checks(
        self, image: np.ndarray, face_image: np.ndarray, face_location: Tuple
    ) -> Dict[str, LivenessCheckResult]:
        # Run the sharpness check first to dynamically adjust noise thresholding
        sharpness_result = check_sharpness(face_image)

        checks = {
            "sharpness": lambda: sharpness_result,
            "symmetry": lambda: check_face_symmetry(image, face_location),
            "texture": lambda: check_skin_texture(face_image),
            "moire": lambda: check_moire_patterns(face_image),
            "depth": lambda: check_depth_variation(face_image),
            # Optionally, add edge-based noise detection
            "edge_noise": lambda: edge_based_noise_detection(face_image, self.logger),
            "blink": lambda: check_eye_blink(image, face_location),
            "size": lambda: check_face_size(face_image),
        }

        results = {}
        futures = {
            check_name: self.executor.submit(self._run_check, check_name, check_func)
            for check_name, check_func in checks.items()
        }

        for check_name, future in futures.items():
            try:
                results[check_name] = future.result(timeout=10)
            except Exception as e:
                results[check_name] = LivenessCheckResult(
                    False, 0.0, {"error": str(e)}, 0.0
                )

        return results

    def _run_check(self, check_name: str, check_func) -> LivenessCheckResult:
        start_time = time.time()
        try:
            result = check_func()
            duration = time.time() - start_time

            if isinstance(result, tuple):
                passed, score, details = result
            else:
                passed = result
                score = 1.0 if passed else 0.0
                details = {}

            return LivenessCheckResult(passed, score, details, duration)

        except Exception as e:
            return LivenessCheckResult(
                False, 0.0, {"error": str(e)}, time.time() - start_time
            )
