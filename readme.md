# Face Recognition and Liveness Detection API

This project provides a FastAPI-based API for face recognition, liveness detection, and image quality assessment, making it suitable for applications like KYC and biometric verification.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/face-comp-api.git
cd face-comp-api
```

### 2. Install Dependencies

#### Using Virtual Environment

Create a virtual environment and activate it:

```bash
python3 -m venv env\nsource env/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirement.txt
```

### 3. Run the Application

To start the FastAPI server, run:

```bash
uvicorn main:app --reload
```

The API will be accessible at `http://127.0.0.1:8000`.

## Docker Setup

To use Docker, follow these steps:

1\. Build the Docker Image:

```bash
docker build -t face-comp-app .
```

2\. Run the Docker Container:

```bash
docker run -p 8000:8000 face-comp-app
```

## Key Endpoints

- **`/compare_faces/`**: Compares two face images to determine a match and checks for liveness.
- **`/check_image_quality/`**: Evaluates whether images meet the required quality criteria for facial recognition.
