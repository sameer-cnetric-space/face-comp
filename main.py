from fastapi import FastAPI, File, UploadFile, HTTPException
import face_recognition
import numpy as np
import cv2
from typing import Dict
from pathlib import Path
from liveCheckGPT4 import LivenessDetector
from liveCheckModules.image_quality import is_image_quality_acceptable


app = FastAPI()


@app.post("/compare_faces/")
async def compare_faces(
    selfie: UploadFile = File(...), document: UploadFile = File(...)
):
    # Check if the files are provided
    if not selfie.filename:
        raise HTTPException(status_code=400, detail="Selfie image is required.")
    if not document.filename:
        raise HTTPException(status_code=400, detail="Document image is required.")

    try:
        # Read the uploaded images as numpy arrays
        selfie_image = np.fromstring(await selfie.read(), np.uint8)
        document_image = np.fromstring(await document.read(), np.uint8)

        # Decode images using OpenCV
        selfie_image = cv2.imdecode(selfie_image, cv2.IMREAD_COLOR)
        document_image = cv2.imdecode(document_image, cv2.IMREAD_COLOR)

        # Load the images using face_recognition for encoding
        selfie_encoding = face_recognition.face_encodings(selfie_image)
        document_encoding = face_recognition.face_encodings(document_image)

        # Check if any faces were found
        if len(selfie_encoding) == 0:
            raise HTTPException(
                status_code=400, detail="No face found in the selfie image."
            )
        if len(document_encoding) == 0:
            raise HTTPException(
                status_code=400, detail="No face found in the document image."
            )

        # Compare the faces and get the distance
        results = face_recognition.compare_faces(
            [selfie_encoding[0]], document_encoding[0]
        )
        face_distance = face_recognition.face_distance(
            [selfie_encoding[0]], document_encoding[0]
        )[0]

        # Calculate confidence score for face match
        match_confidence = calculate_confidence(face_distance)

        # Convert numpy boolean to standard Python boolean for match
        match_result = bool(results[0])

        # Perform liveliness check on the selfie image
        with LivenessDetector() as detector:
            is_live, message, liveness_results = detector.process_image(selfie_image)

        # Ensure results are JSON serializable (convert numpy types)
        for check in liveness_results["checks"]:
            liveness_results["checks"][check]["passed"] = bool(
                liveness_results["checks"][check]["passed"]
            )
            liveness_results["checks"][check]["score"] = float(
                liveness_results["checks"][check]["score"]
            )
            liveness_results["checks"][check]["duration"] = float(
                liveness_results["checks"][check]["duration"]
            )

        # Return JSON response
        return {
            "match": match_result,
            "match_confidence": round(match_confidence, 4),
            "liveliness": is_live,
            "liveliness_details": message,
            "liveliness_results": liveness_results,
        }

    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


def calculate_confidence(face_distance, face_match_threshold=0.6):
    """
    Returns confidence score based on face distance. Lower distance indicates higher confidence.
    """
    if face_distance > face_match_threshold:
        return 0.0
    else:
        range_val = 1.0 - face_match_threshold
        linear_val = (1.0 - face_distance) / (range_val * 2.0)
        return linear_val


@app.post("/check_image_quality/")
async def check_image_quality(
    selfie: UploadFile = File(...), document: UploadFile = File(...)
) -> dict:
    # Check if the files are provided
    if not selfie.filename:
        raise HTTPException(status_code=400, detail="Selfie image is required.")
    if not document.filename:
        raise HTTPException(status_code=400, detail="Document image is required.")

    try:
        # Read the uploaded images as numpy arrays
        selfie_image = np.frombuffer(await selfie.read(), np.uint8)
        document_image = np.frombuffer(await document.read(), np.uint8)

        # Decode images using OpenCV
        selfie_image = cv2.imdecode(selfie_image, cv2.IMREAD_COLOR)
        document_image = cv2.imdecode(document_image, cv2.IMREAD_COLOR)

        # Check quality of both images
        selfie_quality_result = is_image_quality_acceptable(selfie_image)
        document_quality_result = is_image_quality_acceptable(document_image)

        # Prepare response based on quality checks
        response = {
            "selfie_quality": {
                "is_acceptable": selfie_quality_result[0],
                "details": selfie_quality_result[1],
            },
            "document_quality": {
                "is_acceptable": document_quality_result[0],
                "details": document_quality_result[1],
            },
        }

        # Add a message if either image fails the quality check
        if not selfie_quality_result[0] and not document_quality_result[0]:
            response["message"] = (
                "Both images do not meet the required quality criteria."
            )
        elif not selfie_quality_result[0]:
            response["message"] = (
                "Selfie image does not meet the required quality criteria."
            )
        elif not document_quality_result[0]:
            response["message"] = (
                "Document image does not meet the required quality criteria."
            )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
