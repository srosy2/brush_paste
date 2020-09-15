# brush_paste

## train data
Для проверки своей модели, я использовал следущюю методику: я обучался на всех объектах в датасете кроме 1, 1 объект я оставлял для предсказания и так я прошелся по всему датасету, при таком проходе accuracy score:0.93, f1 score:0.81, precision score:0.81, recall score: 0.82, для In-out, для MA f1 score:0.95, precision score:0.95, recall score: 0.95. Можно сделать вывод, что сейсас модель AM предсказывает лучше, чем In-out.
Модель и параметры модели записаны в папке config, лучше всего показывают себя линейные модели, это видно по графику distriburion в папке figure, также в этой папке в pdf файле записаны все объекты, где модель совершает ошибку, также можно найти графики показывающие, распределения зависимостей таргета от параметров

## your data
Для предсказания ваших данных, поместите их в папку Data с назавнием Dataset, ваш датасет должен выглядить аналогично тому, что сейчас находтся в этой папке, после запустите файл main.py, он выдаст excel файл с названием и предсказанием