# Image pour train_model_api
FROM python:3.11 AS train-model-api
WORKDIR /app
COPY src/ ./src/
COPY ./metrics .
COPY ./models .
COPY ./dataset .
COPY dataset.dvc .
COPY metrics.dvc .
COPY models.dvc .
COPY ./.dvc .
COPY secrets.yaml .
COPY requirements.txt .
COPY training_dataset_size.txt .
RUN apt-get update && apt-get install -y libgl1
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "src.train_model_api:app", "--host", "0.0.0.0", "--port", "8000"]

# Image pour predict_api
FROM python:3.11 AS predict-api
WORKDIR /app
COPY src/ ./src/
COPY ./models .
COPY models.dvc .
COPY ./.dvc .
COPY requirements.txt .
COPY secrets.yaml .
RUN apt-get update && apt-get install -y libgl1
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "src.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]

# Image pour streamlit_clinic_app
FROM python:3.11 AS streamlit-clinic-app
WORKDIR /app
COPY src/ ./src/
COPY requirements.txt .
RUN apt-get update && apt-get install -y libgl1
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
EXPOSE 8502
CMD ["streamlit", "run", "src/streamlit/streamlit_clinic_app.py", "--server.address=0.0.0.0", "--server.port=8502"]