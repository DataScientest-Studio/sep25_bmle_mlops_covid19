docker compose down

docker compose build --no-cache

docker compose up -d

docker compose exec airflow-scheduler airflow db migrate

docker compose exec airflow-webserver airflow db migrate

docker compose exec airflow-webserver airflow users create   --username admin   --firstname Admin   --lastname User  --role Admin   --email admin@example.com   --password admin
