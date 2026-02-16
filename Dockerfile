# Image pour train_model_api
FROM python:3.11 AS train_model_api
RUN apt-get update && apt-get install -y libgl1
WORKDIR /app
COPY src/train_model_api.py ./src/train_model_api.py
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
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "src.train_model_api:app", "--host", "0.0.0.0", "--port", "8000"]

# Image pour predict_api
FROM python:3.11 AS predict_api
RUN apt-get update && apt-get install -y libgl1
WORKDIR /app
COPY src/predict_api.py ./src/predict_api.py
COPY src/ ./src/
COPY ./models .
COPY models.dvc .
COPY ./.dvc .
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "src.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]

# Image pour streamlit_clinic_app
#FROM python:3.11-slim AS streamlit_clinic_app
#WORKDIR /app
#COPY ./src/streamlit/streamlit_clinic_app.py .
#COPY requirements.txt .
#RUN pip install --upgrade pip
#RUN pip install --no-cache-dir -r requirements.txt
#CMD ["python", "streamlit_clinic_app.py"]
