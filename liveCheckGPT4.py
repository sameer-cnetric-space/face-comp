from liveCheckModules.liveness_detector import LivenessDetector


def main():
    config = {
        "debug": True,
        "face_detection_model": "hog",
        "cache_results": True,
        "max_workers": 4,
    }

    # Now the context manager works as expected
    with LivenessDetector(config) as detector:
        image_path = "./assets/testdata/thumbnail_Avinash Current Photo.jpg"
        is_live, message, results = detector.process_image(image_path)

        print(f"\nLiveness Detection Results: {'PASS' if is_live else 'FAIL'}")
        print(f"Message: {message}")

        if results.get("checks"):
            for check, result in results["checks"].items():
                print(
                    f"{check}: Passed: {result['passed']} Score: {result['score']:.2f}"
                )


if __name__ == "__main__":
    main()
