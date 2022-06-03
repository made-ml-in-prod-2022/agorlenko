Команды запускаются из текущей директории (online_inference)

Сборка образа:
```shell
docker build -t agorlenko/online_inference:v1 .
```

Загрузка из dockerhub:
```shell
docker pull agorlenko/online_inference:v1
```

Запуск:
```shell
docker run -p 8000:8000 agorlenko/online_inference:v1
```

Запуск утилиты для генерации запросов:
```shell
python -m utils.make_requests -f ../ml_project/tests/data/raw/test_heart_cleveland_upload.csv -p 8000 -n 10
```
где:
- -f - путь к файлу с данными
- -p - порт
- -n - количество запросов
