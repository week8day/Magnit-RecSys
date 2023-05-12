import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from flask import Flask, render_template, request, redirect, url_for
from .config import *

from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise import SVD

np.random.seed(42)

# Создаем логгер и отправляем информацию о запуске
# Важно: логгер в Flask написан на logging, а не loguru,
# времени не было их подружить, так что тут можно пересоздать 
# logger из logging
logger.add(LOG_FOLDER + "log.log")
logger.info("Наш запуск")

# Создаем сервер и убираем кодирование ответа
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  

@app.route("/<task>")
def main(task: str):
    """
    Эта функция вызывается при вызове любой страницы, 
    для которой нет отдельной реализации

    Пример отдельной реализации: add_data
    
    Параметры:
    ----------
    task: str
        имя вызываемой страницы, для API сделаем это и заданием для сервера
    """
    return render_template('index.html', task=task)

@app.route("/add_data", methods=['POST'])
def upload_file():
    """
    Страница на которую перебросит форма из main 
    Здесь происходит загрузка файла на сервер
    """
    def allowed_file(filename):
        """ Проверяем допустимо ли расширение загружаемого файла """
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    # Копируем шаблон ответа для сервера и устанавливаем выполняемую задачу
    answer = ANSWER.copy()
    answer['Задача'] = 'add_data'

    # Проверяем наличие файла в запросе
    if 'file' not in request.files:
        answer['Сообщение'] = 'Нет файла'
        return answer
    file = request.files['file']

    # Проверяем что путь к файлу не пуст
    if file.filename == '':
        answer['Сообщение'] = 'Файл не выбран'
        return answer
    
    # Загружаем
    if file and allowed_file(file.filename):
        # ------- filename = hashlib.md5(file.filename.encode()).hexdigest()
        #file.save(
        #    os.path.join(
        #        UPLOAD_FOLDER, 
        #        filename + file.filename[file.filename.find('.'):]
        #        )
        #    )
        filename = INPUT_FILE_NAME
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        answer['Сообщение'] = 'Файл успешно загружен!'
        answer['Успех'] = True
        answer['Путь'] = filename
        return answer
    else:
        answer['Сообщение'] = 'Файл не загружен'
        return answer
        
@app.route("/show_data", methods=['GET'])
def show_file():
    """
    Страница выводящая содержимое файла
    """
   
    # Копируем шаблон ответа для сервера и устанавливаем выполняемую задачу
    answer = ANSWER.copy()
    answer['Задача'] = 'show_file'

    # Проверяем, что указано имя файла
    if 'path' not in request.args:
        answer['Сообщение'] = 'Не указан путь файла'
        return answer
    file = request.args.get('path') 
    
    # Проверяем, что указан тип файла
    if 'type' not in request.args:
        answer['Сообщение'] = 'Не указан тип файла'
        return answer
    type = request.args.get('type')

    file_path = os.path.join(UPLOAD_FOLDER, file + '.' + type)

    # Проверяем, что файл есть
    if not os.path.exists(file_path):
        answer['Сообщение'] = 'Файл "' + file_path + '"" не существует'
        return answer

    answer['Сообщение'] = 'Файл успешно загружен!'
    answer['Успех'] = True
    
    # Приводим данные в нужный вид
    if type == 'csv':
        answer['Данные'] = pd.read_csv(file_path).to_dict()
        return answer
    else:
        answer['Данные'] = 'Не поддерживаемый тип'
        return answer
    
@app.route("/start", methods=['GET'])
def start_model():
    """
    Обучение модели и формирование рекомендаций
    """

    # Копируем шаблон ответа для сервера и устанавливаем выполняемую задачу
    answer = ANSWER.copy()
    answer['Задача'] = 'start'

    df = pd.read_csv(TRAIN_FOLDER + 'train_joke_df.csv')
    df = df.sort_values(by=['UID', 'JID'])
    df = df.reset_index(drop=True)

    reader = Reader(rating_scale=(-10, 10))

    data = Dataset.load_from_df(df[['UID', 'JID', 'Rating']], reader)

    trainset_data = data.build_full_trainset()

    trainset, testset = train_test_split(data, test_size=0.00001, random_state=42)

    algo = SVD(n_factors=2800, n_epochs=20, biased=True, init_mean=0, init_std_dev=0.019, lr_all=0.0031, reg_all=0.02, random_state=42)
    algo.fit(trainset)

    test = pd.read_csv(UPLOAD_FOLDER + INPUT_FILE_NAME)

    output = test.copy()
    r = []
    for n in test.values:
      s = []
      for p in range(1, 101):
        s.append([p, algo.predict(int(n), p, verbose=False).est])
      s = pd.DataFrame(s, columns=['JID', 'Rating'])
      s = s.sort_values('Rating', ascending=False)
      top_joke_id = list(s[:1]['JID'])[0]
      top_joke_rating = list(s[:1]['Rating'])[0]
      top10 = s[:10]['JID'].values
      top10_str = ' '.join(list(s[:10]['JID'].map(str)))
      r.append([{top_joke_id: top_joke_rating}, list(top10)])

    output['REC'] = r

    output.to_csv(UPLOAD_FOLDER + OUTPUT_FILE_NAME, index=False)

    answer['Сообщение'] = 'Файл рекомендаций успешно сформирован!'
    answer['Успех'] = True
    answer['Путь'] = OUTPUT_FILE_NAME
    return answer