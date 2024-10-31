# Use an official Python image as a base
FROM python:3.9-slim

# Set a working directory in the container
WORKDIR /app

# Copy only the requirements file to leverage Docker cache
COPY requirement.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]