# Product-recommendation-system
**1.1. Задача проекта. **
Создать сервис, выдающий 3 рекомендации товаров для компании-ритейлера по идентификатору пользователя 

**1.2. Бизнес метрика.**
Повышение прибыли от допродаж в интернет-магазине для компании-ритейлера на 20%.

**1.3. Технические метрики.**
Precision@3 - показывает, какую долю из 3 рекомендованных товаров, купил пользователь:

𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛@3=(количество купленных товаров из рекомендованного)/3

Recall@3 - показывает, какая доля из того, что купил пользователь, приходится на рекомендованное:

R𝑒𝑐𝑎𝑙𝑙@3=(количество купленных товаров из рекомендованного)/(количество купленных товаров)

**Модель**
Для модели рекомендательной системы была выбрана матричная факторизация и использована библиотека LightFM c WARP (Weighted Approximate-Rank Pairwise) loss. 
WARP Loss - эта функция потерь лучше других показывает себя в задачах ранжирования. Она работает с тройками (user, positive_item, negative_item) и имеет одну очень важную особенность – выбор негативных примеров происходит не случайно, а таким образом, чтобы выбранные негативные примеры «ломали» текущее ранжирование модели, т.е. были выше, чем позитивный пример.
Пример кода из системы:
![image](https://github.com/sderevyanov/Product-recommendation-system/assets/82756474/ee56454e-6169-4a81-8d19-110bc4cc6c46)

где:
no_components – размерность скрытых вложений признаков;
learning_rate - начальная скорость обучения для расписания обучения адаптивного градиентного спуска;
max_sampled - максимальное количество отрицательных образцов, используемых во время обучения с WARP loss.

**Общая логика выдачи рекомендаций:**
1. Если у пользователя не было транзакции, но был просмотр или добавление в корзину в тренировочной выборке, то ему будут рекомендоваться топ-3 itemid из топ-10 товаров из топ-10 групп свойств. Относительно последнего действия пользователя в тренировочной выборке определяются свойства из топ-10, в которые мог бы входить его последний itemid. Если у пользователя последний itemid не входит ни в одно свойства из топ-10, то ему выдается рекомендация из топ-3 самых просматриваемых itemid в тренировочном датасете;
2. Для новых пользователей или пользователей, которые есть в тестовой выборке, но нет в тренировочной, выдается рекомендация из топ-3 самых просматриваемых itemid в тренировочном датасете;
3. Для пользователей, у которых была транзакция, выдается рекомендация модели, обученной топ-20 купленных товарах на LightFM user-item.

**Метрики полученные на валидационном датасете модели LightFM:**
1. Precision@3 = 15,78 %
Из 3 рекомендуемых товаров пользователи в среднем покупают 15,78 % из них.
2. Recall@3 = 6,19 %
Из 3 рекомендуемых товаров 6,19 %, что купил пользователь, приходится на рекомендованное.
