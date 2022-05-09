# ml project

## train
Для обучения модели нужно выполнить следующую команду:
```shell
python -m ml_project.train.run --config configs/random_forest_config.yaml
```
где configs/random_forest_config.yam - путь к конфигу соответствующей модели. 
При этом обученная модель сохранится в файл, который указан, как output_model_path в конфиге.
Метрики будут сохранены в в файл, указанный как metric_path в конфиге.
Датасет для обучения должен лежать в файле, указанном как input_data_path в конфиге.

