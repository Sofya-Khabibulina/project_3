import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns

#1. Загрузка данных из JSON файлов
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

vacancies_spb = load_json('vacancies_spb.json')
vacancies_chel = load_json('vacancies_chel.json')
vacancies_tol = load_json('vacancies_tol.json')

#2. Добавление меток городов перед объединением
for vacancy in vacancies_spb:
    vacancy['city'] = 'Санкт-Петербург'
for vacancy in vacancies_chel:
    vacancy['city'] = 'Челябинск'
for vacancy in vacancies_tol:
    vacancy['city'] = 'Тольятти'

#3. Объединение данных в один список
all_vacancies = vacancies_spb + vacancies_chel + vacancies_tol

#4. Создание DataFrame
df = pd.DataFrame(all_vacancies)

#5. Очистка данных
df['salary'] = df['salary'].replace(["Не указана", "Не указан", ""], pd.NA)
df['experience'] = df['experience'].replace(["Не указана", "Не указан", ""], pd.NA)
df.dropna(subset=['salary', 'experience'], how='any', inplace=True)

#6. Функция для извлечения зарплат
def extract_salary_range(salary_str):
    if pd.isna(salary_str):
        return None, None

    salary_str = str(salary_str).lower().strip()
    if '$' in salary_str or '€' in salary_str:
        return None, None

    cleaned_str = re.sub(r'[^\d.,–\-—\s]', '', salary_str)
    cleaned_str = cleaned_str.replace(',', '.')
    cleaned_str = re.sub(r'[\s\xa0]', '', cleaned_str)

    match_range = re.search(r'(\d+(?:\.\d+)?)[–\-—](\d+(?:\.\d+)?)', cleaned_str)
    if match_range:
        try:
            min_salary = float(match_range.group(1))
            max_salary = float(match_range.group(2))
            return min_salary, max_salary
        except ValueError:
            pass

    match_single = re.search(r'(\d+(?:\.\d+)?)', cleaned_str)
    if match_single:
        try:
            salary = float(match_single.group(1))
            return salary, salary
        except ValueError:
            pass

    return None, None

#7. Извлечение зарплат
df[['min_salary', 'max_salary']] = df['salary'].apply(lambda x: pd.Series(extract_salary_range(x)))
df.dropna(subset=['min_salary', 'max_salary'], how='any', inplace=True)

#8. Фильтрация и обработка аномалий у зарплаты
df = df[(df['max_salary'] >= 1000) & (df['min_salary'] > 0)]
mask = df['max_salary'] < df['min_salary']
df.loc[mask, ['min_salary', 'max_salary']] = df.loc[mask, ['max_salary', 'min_salary']].values

#9. Создание столбца со средним значением зарплаты для анализа
df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2

#10. Построение диаграммы "ящик с усами"(boxplot)
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
sns.set_palette("pastel")

# Создаем boxplot для средних зарплат по городам
boxplot = sns.boxplot(
    x='city',
    y='avg_salary',
    data=df,
    order=['Санкт-Петербург', 'Челябинск', 'Тольятти'],
    showfliers=True,  #Показываем  выбросы
    width=0.6
)

# Настройка оформления графика
plt.title('Распределение зарплат по городам', fontsize=16)
plt.xlabel('Город', fontsize=14)
plt.ylabel('Средняя зарплата (руб)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Добавляем горизонтальные линии для удобства
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Сохраняем график
plt.tight_layout()
plt.savefig('salary_by_city_boxplot.png', dpi=300)
plt.show()

#11. Дополнительная аналитика
print("\nОписательная статистика по городам:")
print(df.groupby('city')['avg_salary'].describe())

# Сохранение данных в csv формате
df.to_csv('vacancies_processed.csv', index=False, encoding='utf-8')