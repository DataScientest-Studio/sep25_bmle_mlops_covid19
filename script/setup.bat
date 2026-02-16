
docker compose up -d

docker compose exec airflow-webserver airflow users create   --username admin   --firstname Admin   --lastname User  --role Admin   --email admin@example.com   --password admin

docker compose exec airflow-webserver airflow db init