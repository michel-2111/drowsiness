FROM python:3.10-slim

# Install dependencies untuk OpenCV, mediapipe, dll
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender-dev libxext6 gcc ffmpeg

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
