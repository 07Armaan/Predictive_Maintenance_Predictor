FROM python:3.12-slim

WORKDIR /app

# 1. Copy dependencies first
COPY requirements.txt .

# 2. Install all requirements in one step
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest of your app files
COPY . /app

EXPOSE 5000

CMD ["python", "app.py"]
