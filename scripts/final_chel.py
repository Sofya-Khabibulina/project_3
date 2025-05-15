import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import time
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

driver = webdriver.Chrome()

# URL для Челябинска
base_url_chel = "https://chelyabinsk.hh.ru/search/vacancy?order_by=publication_time&search_period=30&items_on_page=100&L_save_area=true&hhtmFrom=vacancy_search_filter&enable_snippets=false&area=104&search_field=name&search_field=company_name&search_field=description&work_format=ON_SITE&work_format=REMOTE&text="
all_vacancies_data = []

# Перебираем страницы
for page_num in range(5):
    url = base_url_chel + str(page_num)

    driver.get(url)
    time.sleep(2)

    vacancies = driver.find_elements(By.CSS_SELECTOR, ".magritte-redesign")  # Главный блок вакансии

    for vacancy in vacancies:
        # Парсинг названия вакансии
        try:
            title = vacancy.find_element(By.CSS_SELECTOR, ".bloko-header-section-2").text

            # Парсинг зарплаты
            try:
                salary = vacancy.find_element(By.CSS_SELECTOR,
                                              "span.magritte-text___pbpft_3-0-33.magritte-text_style-primary___AQ7MW_3-0-33.magritte-text_typography-label-1-regular___pi3R-_3-0-33").text.replace(
                    "&nbsp;", " ")
            except NoSuchElementException:
                salary = "Не указана"

            # Парсинг опыта работы
            try:
                experience = vacancy.find_element(By.CSS_SELECTOR,"span[data-qa^='vacancy-serp__vacancy-work-experience']").text.replace("&nbsp;", " ")
            except NoSuchElementException:
                experience = "Не указан"

            # Парсинг графика работы
            try:
                work_format = vacancy.find_element(By.CSS_SELECTOR, "span[data-qa^='vacancy-label-work-schedule']").text.replace("&nbsp;", " ")
            except NoSuchElementException:
                work_format = "На месте работодателя"


            vacancy_data = {
                "title": title,
                "salary": salary,
                "experience": experience,
                "work_format": work_format

            }
            all_vacancies_data.append(vacancy_data)
        except Exception as e:
            print(f"Ошибка парсинга вакансии: {e}")
            continue

# Закрываем браузер
driver.quit()

# Сохраняем данные в JSON файл
with open('vacancies_chel.json', 'w', encoding='utf-8') as f:
    json.dump(all_vacancies_data, f, ensure_ascii=False, indent=2)

print(f"Собрано {len(all_vacancies_data)} вакансий. Данные сохранены в vacancies_chel.json")

# Преобразовываем JSON файл в датафрейм
with open('vacancies_chel.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)[['title', 'salary', 'experience', 'work_format']]

# Переводим названия колонок на русский язык
df = df.rename(columns={
    'title': 'Название',
    'salary': 'Зарплата',
    'experience': 'Опыт',
    'work_format': 'Режим работы'})

# Составляем список ключевых слов для категорий
service_keywords = ['повар', 'бариста', 'официант', 'ресторан', 'кафе', 'уборщик', 'уборщица', 'пекарь', 'пекарня', 'горничная', 'бармен']
it_keywords = ['программист', 'аналитик', 'разработчик', 'analyst', 'developer', 'гейм', 'game', "Python", 'C++', 'JavaScript', "C#", 'Java', 'IT']
management_keywors = ['менеджер', 'администратор', 'управляющий']
sales_keywords = ['продаж', 'кассир', 'продавец-кассир', 'продавец-консультант', 'продавец', 'sales', "товаровед", "закупк", "кладовщик", "директор магазина"]
med_keywors = ['психолог', 'психиатр', 'терапевт', 'окулист', 'врач', 'фармацевт', 'стоматолог', 'лаборант', 'рентгенолаборант', 'аптека',
               'сиделка', 'массажист', "санитар", "медицинская сестра", "медицинский брат", "медсестра", "медбрат", "косметолог"]
ship_keywords = ['WB', 'OZON', 'Яндекс Маркет', 'доставка', 'заказ', 'ВБ', 'Озон', 'доставщик', 'логист', 'склад', 'курьер']
finance_keywords = ['финанс', 'бухгалтер', "инвест", "экономист"]
call_keywords = ['call', 'оператор', 'чат', 'клиент', 'поддержка']
market_keywords = ['маркет', 'маркетолог', 'реклам']
auto_keywords = ['водитель', 'грузчик', 'автомеханик', 'автослесарь', "автоэлектрик", "автомобил", "мойщик"]
industr_keywords = ['инженер', 'электромонтажник', 'разнорабочий', 'энергетик', 'техник', 'слесар', 'фрезеровщик', "склейщик", "прораб", "технический специалист", "мастер по ремонту", "токарь"]
jur_keywords = ["юрист", "юрисконсульт", "адвокат", "документооборот"]
teach_keywords = ["учитель", "преподаватель", "репетитор"]
housing_keywords = ["агент по недвижимости", "риелтор"]

# Создаем колонку "Категория"
def categorize(title):
    title_lower = title.lower()  # для регистронезависимости
    if any(word in title_lower for word in service_keywords):
        return 'Сфера обслуживания'
    elif any(word in title_lower for word in it_keywords):
        return 'IT'
    elif any(word in title_lower for word in management_keywors):
        return 'Менеджмент'
    elif any(word in title_lower for word in sales_keywords):
        return 'Продажи'
    elif any(word in title_lower for word in med_keywors):
        return 'Медицина'
    elif any(word in title_lower for word in ship_keywords):
        return 'Сфера доставки'
    elif any(word in title_lower for word in finance_keywords):
        return 'Финансы/бухгалтерия'
    elif any(word in title_lower for word in call_keywords):
        return 'Поддержка клиентов'
    elif any(word in title_lower for word in market_keywords):
        return 'Маркетинг и реклама'
    elif any(word in title_lower for word in auto_keywords):
        return 'Автомобильная сфера'
    elif any(word in title_lower for word in industr_keywords):
        return 'Промышленность'
    elif any(word in title_lower for word in jur_keywords):
        return 'Юриспруденция'
    elif any(word in title_lower for word in teach_keywords):
        return 'Педагогика и преподавание'
    elif any(word in title_lower for word in housing_keywords):
        return 'Недвижимость'
    else:
        return 'Другое'

df['Категория'] = df['Название'].apply(categorize)

# Преобразование типов данных и очистка (и EDA)

# Заменяем текстовые обозначения пропусков в нужных столбцах
df['Зарплата'] = df['Зарплата'].replace(["Не указана", "Не указан", ""], pd.NA)
df['Опыт'] = df['Опыт'].replace(["Не указана", "Не указан", ""], pd.NA)

print("\nКоличество пропусков до очистки:")
print(df.isnull().sum())

# Удаляем строки, где есть пропуски в salary или experience
df.dropna(subset=['Зарплата', 'Опыт'], how='any', inplace=True)

def extract_salary_range(salary_str):
    """
    Извлекает диапазон зарплат из строки и возвращает его в числовом формате (min, max).
    Если строка содержит только одно число, то min и max зарплаты равны этому числу.
    Если строка не содержит ни диапазон, ни одно число, возвращает None.
    """
    if pd.isna(salary_str):
        return None, None

    salary_str = str(salary_str).lower().strip()

    # Определяем валюту (убираем доллары и евро)
    if '$' in salary_str :
        return None, None
    if '€' in salary_str :
        return None, None

    # Удаляем все нецифровые символы, кроме точек, запятых и разделителей диапазона
    cleaned_str = re.sub(r'[^\d.,–\-—\s]', '', salary_str)

    # Заменяем запятые на точки (для десятичных разделителей)
    cleaned_str = cleaned_str.replace(',', '.')

    # Заменяем все пробелы (включая неразрывные) и другие разделители тысяч
    cleaned_str = re.sub(r'[\s\xa0]', '', cleaned_str)

    # Исправленное регулярное выражение (добавлена закрывающая скобка)
    match_range = re.search(r'(\d+(?:\.\d+)?)[–\-—](\d+(?:\.\d+)?)', cleaned_str)
    if match_range:
        try:
            min_salary = float(match_range.group(1))
            max_salary = float(match_range.group(2))
            return min_salary, max_salary
        except ValueError:
            pass

    # Если нет диапазона, ищем одиночное число
    match_single = re.search(r'(\d+(?:\.\d+)?)', cleaned_str)
    if match_single:
        try:
            salary = float(match_single.group(1))
            return salary, salary
        except ValueError:
            pass

    return None, None


# Применяем функцию к столбцу salary и создаем новые столбцы
df[['min_salary', 'max_salary']] = df['Зарплата'].apply(lambda x: pd.Series(extract_salary_range(x)))

# Фильтрация
df = df[df['max_salary'] >= 1000]
df = df[df['min_salary'] > 0]

print("\nКоличество пропусков до очистки:")
print(df.isnull().sum())

# Удаляем строки, где не удалось извлечь какую-либо зарплату
df.dropna(subset=['min_salary', 'max_salary'], how='any', inplace=True)


# Обработка аномалий
# Инверсия min и max, если max < min
mask = df['max_salary'] < df['min_salary']
df.loc[mask, ['min_salary', 'max_salary']] = df.loc[mask, ['max_salary', 'min_salary']].values

# Выводим несколько первых строк
print("Первые строки:")
print(df.head())

# Проверяем результат
print("\nСтатистика после очистки:")
print(f"Осталось строк: {len(df)}")
print("\nКоличество пропусков:")
print(df.isnull().sum())

# 5. Минимальный EDA для всех переменных
print("Описание типов переменных:\n", df.dtypes)
print("\nОписательные статистики:\n", df.describe())
low_salaries = df[df['min_salary'] == 300]
print(low_salaries[['Зарплата', 'min_salary', 'max_salary']])
print("\nУникальные значения в столбце 'title':\n", df['Название'].unique())
print("\nНаиболее частые значения в столбце 'title':\n", df['Название'].value_counts())
print("\nНаиболее частые значения в столбце 'experience':\n", df['Опыт'].value_counts())
print("\nНаиболее частые значения в столбце 'work_format':\n", df['Режим работы'].value_counts())

df.to_csv('vacancies_chel.csv', index=False, encoding='utf-8')

# Открываем csv файл для работы над визуализацией
vacanciesplots = pd.read_csv('vacancies_chel.csv', sep=',', encoding='utf-8')

# Круговая диаграмма по требованиям работы
fig, ax = plt.subplots(figsize=(12, 8))

# Получаем распределение
requirements_counts = vacanciesplots['Опыт'].value_counts()

# Строим круговую диаграмму с настройками
requirements_counts.plot(kind='pie',
                    ax=ax,
                    autopct='%1.1f%%',    # Проценты
                    startangle=90,       # Начало отсчета углов
                    counterclock=False,   # Направление размещения
                    pctdistance=0.8,     # Радиус для процентов
                    labeldistance=1.05,   # Радиус для подписей
                    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},  # Границы секторов
                    textprops={'fontsize': 10})  # Размер шрифта

# Настройки графика
ax.set_title("Распределение вакансий по требуемому опыту", fontsize=14, pad=20)
ax.set_ylabel('')

# Убираем лишние границы
plt.box(False)

# Добавляем легенду справа
plt.legend(labels=requirements_counts.index,
           bbox_to_anchor=(1.05, 0.8),
           loc='upper left')

plt.show()

#Группированная столбчатая диаграмма для двух переменных
# Группировка данных
grouped = vacanciesplots.groupby(['Категория', 'Опыт']).size().unstack()

# Настройка графика
plt.figure(figsize=(14, 8))
grouped.plot(kind='bar', stacked=False, colormap='tab20', alpha=0.9, edgecolor='black')

# Подписи и заголовок
plt.title('Распределение вакансий по категориям и опыту работы', fontsize=14)
plt.xlabel('Категория', fontsize=12)
plt.ylabel('Количество вакансий', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Опыт', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

#Scatterplot (диаграмма рассеяния) для двух переменных

# Создание числовых кодов для категорий
cat_codes, cat_labels = pd.factorize(vacanciesplots['Категория'])
work_codes, work_labels = pd.factorize(vacanciesplots['Режим работы'])

# Добавление  шума
np.random.seed(42)
jitter_cat = cat_codes + np.random.uniform(-0.3, 0.3, size=len(cat_codes))
jitter_work = work_codes + np.random.uniform(-0.3, 0.3, size=len(work_codes))

# Создание графика
plt.figure(figsize=(12, 8))
plt.scatter(jitter_cat, jitter_work, alpha=0.5)

# Настройка осей
plt.xticks(ticks=np.arange(len(cat_labels)), labels=cat_labels, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(work_labels)), labels=work_labels)

# Подписи и заголовок
plt.xlabel('Категория')
plt.ylabel('Режим работы')
plt.title('Распределение вакансий по категориям и режимам работы')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()