# Нейронная сеть для определения персональной информации

# Установка зависимостей

Python 3.12+
```
pip install -r .\requirements.txt
pip install -r .\requirements_torch.txt
```

# Обучение модели

Для обучения модели необходимо скачать и распаковать датасет 
https://storage.yandexcloud.net/natasha-nerus/data/nerus_lenta.conllu.gz

Обучение организовано через Jupyter Notebook
model_training_bilstm_crf.ipynb - Обучение и валиадация модели BiLETM+CRF
model_training_bert_crf.ipynb - Обучение и валидация модели BERT+CRF 

В файл сохраняется только BERT+CRF, т.к. BiLETM+CRF показывает недостаточную эффективность.

# Обученная модель 
https://github.com/IstariTM/pii_detecting_and_masking_model/releases/tag/v0.0.1

# Использование модели для маскирования данных

Обученнную модель можно использовать с помощью GUI, реализованного через tkinter.
Запуск 
```
py ./PiiMasker.py
```
