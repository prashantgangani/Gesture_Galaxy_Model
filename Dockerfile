FROM python:3.10

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Use Flask's built-in server for simplicity (or gunicorn if preferred)
CMD ["python", "app.py"]