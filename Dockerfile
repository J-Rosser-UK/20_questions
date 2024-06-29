FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install torch==2.2.0 --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
