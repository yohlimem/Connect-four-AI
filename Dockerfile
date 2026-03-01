# Start with a lightweight Python 3.10 image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy requirements first to leverage Docker caching
COPY ./requirements.txt /code/requirements.txt

# Install dependencies (no cache to keep image small)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the entire project directory into the container
# This includes main.py, your .pth model, the 'web' folder, and all helper .py files
COPY . /code

# Create the directory for the SQLite database if it doesn't exist
# Note: Data here will reset if the Space restarts (ephemeral storage)
RUN mkdir -p /code/data

# Command to run the application on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]