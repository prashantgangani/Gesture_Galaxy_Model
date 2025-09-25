FROM python:3.10

WORKDIR /app
COPY . /app

# Install OpenCV dependencies (use libgl1 instead of libgl1-mesa-glx)
RUN apt-get update && apt-get install -y libgl1

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "app.py"]