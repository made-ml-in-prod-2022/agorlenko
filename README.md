# ml project

## train
Для обучения модели нужно выполнить следующую команду:
```shell
python -m ml_project.train.run --config configs/random_forest_config.yaml
```
где configs/random_forest_config.yaml - путь к конфигу соответствующей модели. 
При этом обученная модель сохранится в файл, который указан, как output_model_path в конфиге.
Метрики будут сохранены в в файл, указанный как metric_path в конфиге.
Датасет для обучения должен лежать в файле, указанном как input_data_path в конфиге.

## predict
Для запуска предсказания нужно выполнить следующую команду:
```shell
python -m ml_project.predict.run --config configs/train_config.yaml --test_df test_heart_cleveland_upload.csv --output_file out
```
где configs/random_forest_config.yaml - путь к конфигу соответствующей модели.
В параметре --test_df указываем путь к датасету, для которого надо сгенерировать предсказание.
В параметре --output_file указывает путь к файлу, где нужно сохранить предсказание. 
Если значение не указано, то результат будет выведен в stdout.

## Запуск тестов
Для запуска тестов надо выполнить команду:
```shell
pytest ml_project/tests/
```
