import logging
import os

import joblib
from datetime import datetime
import json

import numpy as np
import pandas as pd
import math
import scipy
from fastapi import FastAPI
from sklearn.model_selection import train_test_split
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse


from fastapi.responses import JSONResponse
from prometheus_client import Gauge, CollectorRegistry, push_to_gateway

from .config import path_logs, path_dataset, path_models
from .schema import ModelTrain, ModelOutp, ForecastOutp
import lightfm
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

import warnings
warnings.filterwarnings("ignore")

# Создадим логирование
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
path = path_logs()
filename = os.path.join(path, 'API_log_' +
                        str(datetime.now().year) +
                        str(datetime.now().strftime("%m")) +
                        str(datetime.now().strftime("%d")) +
                        '.log')
fh = logging.FileHandler(filename=filename)
formatter = logging.Formatter(
    "%(asctime)s - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
)
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)  # Exporting logs to the screen
logger.addHandler(fh)  # Exporting logs to a file

# Сделаем оформление UI для FastApi
app = FastAPI(title='Product recommendation system',
              description='Products recommendation for userid',
              version='0.0.1',
              contact={
                  "name": "Sergey Derevyanov",
                  "email": "derevyanov@mail.ru"},
              license_info={
                  "name": "Apache 2.0",
                  "url": "https://www.apache.org/licenses/LICENSE-2.0.html"},
              openapi_tags=[{"name": "check", "description": "Check html response."},
                            {"name": "train", "description": "Operations with training models."},
                            {"name": "forecast", "description": "Operations with forecasting"}])

# Включим CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------------------------------
# Зададим пути для сохранения моделей и моделей нормализации данных

path_m = path_models()
path_d = path_dataset()
registry = CollectorRegistry()
pr = Gauge('LightFM_metric_precision_at_3', 'Metric precision_at_3 after train LightFM model', registry=registry)
rec = Gauge('LightFM_metric_recall_at_3', 'Metric recall_at_3 after train LightFM model', registry=registry)

# ---------------------------------------------------------------------------------------------------


def get_key(dictionary, value):
    """
    Функция получения ключа словаря по значению.
    """
    for k, v in dictionary.items():
        if v == value:
            return k


@app.get("/", name='Check HTMLResponse', tags=["check"])
def root():
    return HTMLResponse("<b>Product recommendation system API</b>")


@app.post('/model_train/', response_model=ModelOutp, name='Train model', tags=["train"])
async def train_model(data_js: ModelTrain):
    """
    Создание модели машинного обучения с помощью алгоритма CatBoostRegressor.
    """
    global path_m, path_d, pr, registry

    start_time = datetime.now()

    # Загрузим данные из файла .json
    try:
        data = json.loads(data_js.json())
        logger.info('Данные загружены')
    except:
        logger.error('Данные неверного формата, ошибка в исходном файле')
        logger.info('Завершено')
        return {'Training_time': str(datetime.now() - start_time),
                'Status': str('Error'),
                'Message': str('Данные неверного формата, ошибка в исходном файле'),
                'accuracy_score': '0',
                'f1_score': '0'}

    # Создадим датафрейм
    events_data = pd.DataFrame(data['Data'])
    events_data = events_data.sort_values('timestamp')
    events_data['time'] = events_data['timestamp'].apply(lambda x: datetime.fromtimestamp(float(x / 1000)))
    events_data['transactionid_cnt'] = events_data['transactionid'].apply(lambda x: 0 if math.isnan(x) else 1)
    split_time = events_data['time'].max() - pd.Timedelta("31 days")
    train_events = events_data.loc[events_data['time'] < split_time]
    test_events = events_data.loc[events_data['time'] >= split_time]

    # train_events, test_events = train_test_split(events_data, test_size=0.2, shuffle=False)

    top_transact = train_events['itemid'].loc[train_events['event'] == 'transaction'].value_counts().index[
                   :20].values.tolist()

    # Далее для постороения модели возьмем только строки, где покупка была осуществлена
    train_events_model = train_events.loc[(train_events['event'] == 'transaction') &
                                          (train_events['itemid'].isin(top_transact))]
    test_events_model = test_events.loc[(test_events['event'] == 'transaction') &
                                        (test_events['visitorid'].isin(train_events_model.visitorid.unique()))]

    train_events_model = train_events_model.drop_duplicates(subset=['visitorid', 'itemid'])
    test_events_model = test_events_model.drop_duplicates(subset=['visitorid', 'itemid'])

    # Создадим отдельный датасет с нужными нам ячейками
    train_events_lfm = train_events_model[['timestamp', 'visitorid',
                                           'itemid', 'transactionid_cnt']].reset_index(drop=True)
    test_events_lfm = test_events_model[['timestamp', 'visitorid',
                                         'itemid', 'transactionid_cnt']].reset_index(drop=True)

    train_events_lfm = train_events_lfm.rename(
        columns={'visitorid': 'user_id', 'itemid': 'item_id', 'transactionid_cnt': 'buy'})
    test_events_lfm = test_events_lfm.rename(
        columns={'visitorid': 'user_id', 'itemid': 'item_id', 'transactionid_cnt': 'buy'})

    # Pivoting TRAIN
    train_pivot = pd.pivot_table(train_events_lfm,
                                 index='user_id',
                                 columns='item_id',
                                 values='buy')

    # Pivoting TEST
    test_pivot = pd.pivot_table(test_events_lfm,
                                index='user_id',
                                columns='item_id',
                                values='buy')

    # Создадим сводную таблицу из таблицы events_df, заполнив её ячейки нулями.
    # Для тех товаров, которые пользователь купил, будут равны 0, для остальных — пропуску.
    events_shell = pd.concat([train_events_lfm, test_events_lfm])
    shell = pd.pivot_table(
        events_shell,
        index='user_id',
        columns="item_id",
        values="buy",
        aggfunc=lambda x: 0
    )

    # Получим тренировочную и тестовую user-item таблицы, сложив таблицу shell с
    # соответствующими таблицами train_pivot и test_pivot.
    # 0 — если пользователь не покупал товар;
    # 1 — если пользователь покупал товар.
    train_pivot = shell + train_pivot
    test_pivot = shell + test_pivot
    train_pivot = train_pivot.fillna(0)
    test_pivot = test_pivot.fillna(0)

    # Получим разреженные матрицы, используется функция csr_matrix() из модуля sparse библиотеки scipy:
    train_pivot_sparse = scipy.sparse.csr_matrix(train_pivot.values)
    test_pivot_sparse = scipy.sparse.csr_matrix(test_pivot.values)

    # Зададим дату старта обучения
    logger.info('Запуск обучения модели.')
    t_train_start = datetime.now()

    # Обучим модель на LightFM
    model_lfm = LightFM(no_components=10, loss='warp', random_state=42, learning_rate=0.03, max_sampled=10)
    model_lfm.fit(train_pivot_sparse, epochs=10)

    # Посчитаем время затраченное на обучение
    time_fit_all = datetime.now() - t_train_start
    logger.info(f'Обучение модели на всех данных завершено. Время обучения: {time_fit_all}')

    # Рассчитаем средний precision для топ 3 рекомендуемых товаров по всей тестовой выборке
    map_at3 = precision_at_k(model_lfm, test_pivot_sparse, k=3).mean()
    logger.info('Mean Average Precision at 3: {:.10f}%'.format(map_at3 * 100))

    # Рассчитаем средний Recall для топ 3 рекомендуемых товаров по всей тестовой выборке
    rec_at3 = recall_at_k(model_lfm, test_pivot_sparse, k=3).mean()
    logger.info('Mean Average Recall at 3: {:.10f}%'.format(rec_at3 * 100))

    logger.info('Сохраняем обученную модель и датасет.')
    # Сохраним модель
    filename_m = os.path.join(path_m, 'Recommendations_LightFM.joblib')
    joblib.dump(model_lfm, open(filename_m, 'wb'))
    # Сохраним train_pivot
    filename_tp = os.path.join(path_d, 'Recommendations_train_pivot.joblib')
    joblib.dump(train_pivot, open(filename_tp, 'wb'))
    # Сохраним events_data
    filename_d = os.path.join(path_d, 'Recommendations_events_df.joblib')
    joblib.dump(events_data, open(filename_d, 'wb'))
    logger.info('Модель и датасет сохранены.')

    logger.info('Передаем метрики push_to_gateway.')
    # Передадим метрики в prometheus через push_to_gateway
    try:
        pr.set(map_at3)  # Set to a given value
        push_to_gateway('localhost:9091', job='batchA', registry=registry)
        logger.info('Метрика precision_at_k передана push_to_gateway.')
    except:
        logger.info('Метрика precision_at_k не передана push_to_gateway.')
        pass
    try:
        rec.set(rec_at3)  # Set to a given value
        push_to_gateway('localhost:9091', job='batchB', registry=registry)
        logger.info('Метрика recall_at_k передана push_to_gateway.')
    except:
        logger.info('Метрика recall_at_k не передана push_to_gateway.')
        pass

    logger.info('Обучение модели завершено.')

    return {'Training_time': str(datetime.now() - start_time),
            'Status': str('Success'),
            'Message': str('Обучение модели рекомендация товаров на LightFM прошло успешно.'),
            'precision_at_3': str(f"{map_at3:0.5f}"),
            'recall_at_3': str(f"{rec_at3:0.5f}")
            }


@app.get("/get_userid_dict", name='Get userid dictionary', tags=["forecast"])
def get_userid_dict():
    # Загрузим датасет из хранилища, на котором обучалась модель
    try:
        filename_d = os.path.join(path_d, "Recommendations_events_df.joblib")
        with open(filename_d, 'rb') as d:
            events_data = joblib.load(d)
    except FileNotFoundError:
        logger.error('Датасет {0} не существует.'.format("Recommendations_events_df.joblib"))
        logger.info('Завершено')
        return JSONResponse(status_code=418,
                            content=str('Датасет {0} не существует.'.format("Recommendations_events_df.joblib")))
    logger.info('Датасет загружен')

    # Создадим словарь с соотношением индекса и реального значения user_id
    user_id_dict = pd.Series(events_data.visitorid).to_dict()
    logger.info('Созданы словарь с парой user_index - user_id')
    return user_id_dict


@app.get("/recommendation", response_model=ForecastOutp, name='Product recommendation', tags=["forecast"])
async def recommendation(user_id: int):
    global path_m, path_d

    # Зафиксируем время начала выполнения скрипта
    start_time = datetime.now()

    # Проверим корректно ли введено значение user_id
    try:
        if type(user_id) != int:
            raise ValueError()
    except ValueError:
        logger.error('Не верный формат введенных данных: {0}, '
                     'необходимо ввести число в формате int'.format(type(user_id)))
        logger.info('Завершено')
        return JSONResponse(status_code=418,
                            content=str('Не верный формат введенных данных: {0}, необходимо '
                                        'ввести число в формате int'.format(type(user_id))))

    # Загрузим модель из хранилища
    try:
        filename_m = os.path.join(path_m, "Recommendations_LightFM.joblib")
        with open(filename_m, 'rb') as f:
            model = joblib.load(f)
    except FileNotFoundError:
        logger.error('Модель {0} не существует.'.format("Recommendations_LightFM.joblib"))
        logger.info('Завершено')
        return JSONResponse(status_code=418,
                            content=str('Модель {0} не существует.'.format("Recommendations_LightFM.joblib")))
    logger.info('Модель загружена')

    # Загрузим train_pivot_initial из хранилища, на котором обучалась модель
    try:
        filename_tp = os.path.join(path_d, "Recommendations_train_pivot.joblib")
        with open(filename_tp, 'rb') as tp:
            train_pivot_initial = joblib.load(tp)
    except FileNotFoundError:
        logger.error('Датасет {0} не существует.'.format("Recommendations_train_pivot.joblib"))
        logger.info('Завершено')
        return JSONResponse(status_code=418,
                            content=str('Датасет {0} не существует.'.format("Recommendations_train_pivot.joblib")))
    logger.info('Датасет train_pivot загружена')

    # Загрузим датасет из хранилища, на котором обучалась модель
    try:
        filename_d = os.path.join(path_d, "Recommendations_events_df.joblib")
        with open(filename_d, 'rb') as pf:
            events_data = joblib.load(pf)
    except FileNotFoundError:
        logger.error('Датасет {0} не существует.'.format("Recommendations_events_df.joblib"))
        logger.info('Завершено')
        return JSONResponse(status_code=418,
                            content=str('Датасет {0} не существует.'.format("Recommendations_events_df.joblib")))
    logger.info('Датасет events_data загружен')

    # Загрузим датасет properties_filtered из хранилища
    try:
        filename_pf = os.path.join(path_d, "Recommendations_properties_filtered_df.joblib")
        with open(filename_pf, 'rb') as d:
            properties_filtered = joblib.load(d)
    except FileNotFoundError:
        logger.error('Датасет {0} не существует.'.format("Recommendations_properties_filtered_df.joblib"))
        logger.info('Завершено')
        return JSONResponse(status_code=418,
                            content=str('Датасет {0} не существует.'.format(
                                "Recommendations_properties_filtered_df.joblib")))
    logger.info('Датасет properties_filtered загружен')

    train_pivot = train_pivot_initial.loc[train_pivot_initial.index != user_id]
    logger.info('Созданы данные для проноза')

    train_pivot_sparse = scipy.sparse.csr_matrix(train_pivot.values)
    logger.info('Созданы разреженные матрицы для проноза')

    # Создадим словарь с соотношением индекса и реального значения user_id
    user_id_dict = pd.Series(train_pivot_initial.index).to_dict()
    logger.info('Созданы словать с парой user_undex - user_id')
    # logger.info(user_id_dict)

    # Проверим есть ли наш UserID в обученной модели
    try:
        if user_id not in events_data.visitorid.unique():
            raise ValueError()
    except ValueError:
        logger.error('Введенный UserID отсутствует в модели для рекомендаций товаров: {}'.format(user_id))
        logger.info('Завершено')
        return JSONResponse(status_code=418,
                            content=str('Введенный UserID отсутствует в модели '
                                        'для рекомендаций товаров: {}'.format(user_id)))
    logger.info('Для пользователя существуют рекомендации')

    # Создадим копию загруженного датасета events_data
    events = events_data.copy()
    # Создадим фиктивный столбец с признаком транзакции: 1- транзация была, 0 - нет
    events['transactionid_cnt'] = events['transactionid'].apply(lambda x: 0 if math.isnan(x) else 1)
    # Отсортируем датасет по столбцу timestamp
    events = events.sort_values('timestamp')
    events['time'] = events['timestamp'].apply(lambda x: datetime.fromtimestamp(float(x / 1000)))
    logger.info('Датасет отсортирован по timestamp.')

    # Разделим датасет на тренировочную и тестовую выборку
    # train_events, test_events = train_test_split(events, test_size=0.2, shuffle=False)
    split_time = events['time'].max() - pd.Timedelta("31 days")
    train_events = events.loc[events['time'] < split_time]
    # test_events = events.loc[events['time'] >= split_time]
    logger.info('Разделение датасет на тренировочную и тестовую выборку, прошло успешно.')

    # Создадим вспомогательный датасет сгруппированный по property с подсчетом каждого itemid в каждом property
    properties_filtered['itemid_cnt'] = 1
    items_cnt = properties_filtered[['property', 'itemid', 'itemid_cnt']].groupby([
        'property', 'itemid'])['itemid_cnt'].sum().reset_index()
    logger.info('Вспомогательный датасет сгруппированный по property с подсчетом каждого itemid '
                'в каждом property, создан.')

    # Создадим словарь с топ 10 itemid для топ 10 свойств товаров
    top10_4_top_prop_dict = {}
    for i in set(np.unique(items_cnt.property)):
        if items_cnt.loc[items_cnt['property'] == str(i), 'itemid_cnt'].max() == 1:
            top10_4_top_prop_dict[i] = np.random.choice(items_cnt.loc[items_cnt['property'] == str(i), 'itemid'],
                                                        size=10)
        else:
            top10_4_top_prop_dict[i] = items_cnt.loc[items_cnt['property'] == str(i)].sort_values(
                'itemid_cnt', ascending=False)[:10]['itemid'].tolist()
    logger.info('Словарь с топ 10 itemid для топ 10 свойств товаров, создан.')
    # Создадим множество из уникальных visitorid в тренеровочной выборке у которых не было транзакций
    user_recs_train = train_events.copy()
    user_recs_train = user_recs_train.drop_duplicates(subset=['visitorid', 'event', 'itemid'])
    users_wot_dict_train = (set(user_recs_train.visitorid.unique()) -
                            set(user_recs_train[user_recs_train['event'] == 'transaction'].visitorid.unique()))
    logger.info('Множество из уникальных visitorid в тренеровочной выборке у которых не было транзакций, создано.')
    # user_recs_test = test_events.copy()
    # user_recs_test = user_recs_test.drop_duplicates(subset=['visitorid', 'event', 'itemid'])

    # users_wot_dict_test = (set(user_recs_test.visitorid.unique()) -
    #                        set(user_recs_test[user_recs_test['event'] == 'transaction'].visitorid.unique()))

    # Блок прогнозирования
    #  - Если у пользователя не было транзакции, но был просмотр или добавление в корзину в тренеровчной выборке, то
    # ему будут рекомендоваться itemid из топ 10 товаров из топ 10 групп свойств в рандомном порядке. Относительно
    # последнего действия пользователя в тренеровочной выборке определяются свойства из топ 10, в которые мог бы входить
    # его последний itemid. Если у пользователя последний itemid не входит ни в одно свойства из топ 10, то ему
    # выдается рекомендация из топ 3 самых просматриваемых itemid в тренировочном датасете
    # - Для новых пользователей или пользователей, которые есть в тестовой выборке, но нет в тренировочной,
    # выдается рекомендация из топ 3 самых просматриваемых itemid в тренировочном датасете
    # - Для пользователей у которых была транзакция выдается рекомендация модели обученной на LightFM user-item
    logger.info('Старт прогноза проноза')
    if user_id in users_wot_dict_train:
        item_user = train_events.loc[(train_events['visitorid'] == user_id)].sort_values(
            'timestamp')[-1:]['itemid'].values[0]
        # print('itemitem', itemitem)
        item_prop = properties_filtered.loc[properties_filtered['itemid'] == item_user, 'property'].unique().tolist()
        # print('item_prop', item_prop)
        if len(item_prop) > 0:
            # result = []
            # for i in item_prop:
            #     result.append(top10_4_top_prop_dict[i][0])
            # results = np.unique(np.random.choice(result, size=3)).tolist()
            results = properties_filtered.loc[
                          properties_filtered['property'].isin(item_prop), 'itemid'].value_counts().index[:3].tolist()
            # print('results', results)
            # print('длина:', len(results))
            if len(results) == 2:
                results.append(int(train_events['itemid'].value_counts().index[2]))
            elif len(results) == 1:
                results.append(int(train_events['itemid'].value_counts().index[1]))
                results.append(int(train_events['itemid'].value_counts().index[2]))
            elif len(results) == 0:
                results.append(int(train_events['itemid'].value_counts().index[0]))
                results.append(int(train_events['itemid'].value_counts().index[1]))
                results.append(int(train_events['itemid'].value_counts().index[2]))
        else:
            results = train_events['itemid'].value_counts().index[:3].values.tolist()
    elif user_id in user_id_dict.values():
        user_id = get_key(user_id_dict, user_id)
        unique_items = np.array(train_pivot.columns)
        item_ids = np.arange(0, train_pivot_sparse.shape[1])
        list_pred = model.predict(user_id, item_ids)
        recommendations_ids = np.argsort(-list_pred)[:3]
        recommendations = unique_items[recommendations_ids].tolist()
        results = recommendations
    else:
        results = train_events['itemid'].value_counts().index[:3].values.tolist()
    results = list(results)
    logger.info('Results: {}'.format(results))
    logger.info('Results type: {}'.format(type(results)))
    logger.info('Прогноз выполнен успешно')

    result_recom = dict(recommendation_1=results[0],
                        recommendation_2=results[1],
                        recommendation_3=results[2])

    return {'Forecast_time': str(datetime.now() - start_time),
            'Status': str('Прогноз выполнен успешно'),
            'Recommendations': result_recom}
