FROM apache/airflow:2.11.0-python3.12
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY --chown=airflow:airflow . .

CMD ["python", "ml_pipeline.py", "--acidity-path", "winequality-red-acidity.csv", "--other-path", "winequality-red-other.csv", "--output-dir", "results"]

