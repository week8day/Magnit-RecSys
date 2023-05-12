# ------------------------------------ #
#   Здесь заданны параметры сервера    #
# ------------------------------------ #

# Путь до папки с данными
UPLOAD_FOLDER = 'my_project/data/'

# Путь до папки с логами
LOG_FOLDER = 'my_project/log/'

# Файлы которые допустимы к загрузке
ALLOWED_EXTENSIONS = {'csv'}

# Шаблон ответа сервера
ANSWER = {
    "Успех": False,
    "Задача": "",
    "Сообщение": ""
}

TRAIN_FOLDER = 'my_project/train/'
INPUT_FILE_NAME = 'test.csv'
OUTPUT_FILE_NAME = 'output.csv'