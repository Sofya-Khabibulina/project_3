import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# Настройка страницы
st.set_page_config(
    page_title="Анализ вакансий",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Загрузка данных
@st.cache_data
def load_data():
    spb = pd.read_csv('data/vacancies_spb.csv')
    chel = pd.read_csv('data/vacancies_chel.csv')
    tol = pd.read_csv('data/vacancies_tol.csv')

    # Добавляем метки городов
    spb['Город'] = 'Санкт-Петербург'
    chel['Город'] = 'Челябинск'
    tol['Город'] = 'Тольятти'

    # Объединяем и сбрасываем индекс
    full_df = pd.concat([spb, chel, tol], ignore_index=True)
    return full_df

df = load_data()

# Навигация
st.sidebar.title("Навигация")
section = st.sidebar.radio(
    "Выберите раздел:",
    ["Главная", "Данные", "EDA", "Тренды", "Выводы"]
)

# Главная страница
if section == "Главная":
    st.title('Анализ вакансий в трех городах России 🔍')
    st.header("О проекте")
    st.markdown("""
    ### Источник данных: hh.ru
    Наш проект представляет собой анализ тенденций на рынке труда трёх городов России. 
    Цель проекта: проанализировать, вакансии в каких сферах являются наиболее востребованными, какие профессии являются наиболее востребованными в трех городах России, разных по числу населения (города: Санкт-Петербург, Челябинск, Тольятти) за последний месяц на основе базы данных hh.ru. 
    Помимо этого мы сравниваем заработные платы, требуемый опыт работы и режим работы, которые соответствуют вакансиям, и выявляем закономерности и различия в данных показателях по каждому городу. 
    Мы собираем данные примерно о 500 вакансиях по каждому городу, сравниваем среднюю заработную плату по каждому городу, сравниваем взаимосвязи между сферами и требованиями по каждой вакансии и так далее.
 
    """)

    # Показываем пример данных
    st.subheader("Пример данных")
    st.dataframe(df.sample(5), use_container_width=True)

# Раздел с данными
elif section == "Данные":
    st.header("Исходные данные 📖")

    # Создаем колонки для фильтров
    col1, col2, col3 = st.columns(3)

    with col1:
        # Фильтр по городу
        selected_city = st.selectbox(
            "Выберите город",
            ["Все"] + list(df['Город'].unique())
        )

    with col2:
        # Фильтр по режиму работы
        work_modes = ["Все"] + list(df['Режим работы'].dropna().unique())
        selected_mode = st.selectbox(
            "Выберите режим работы",
            work_modes
        )

    with col3:
        # Фильтр по опыту работы
        experience_levels = ["Все"] + list(df['Опыт'].dropna().unique())
        selected_exp = st.selectbox(
            "Выберите опыт работы",
            experience_levels
        )

    # Применяем фильтры
    city_df = df.copy()

    if selected_city != "Все":
        city_df = city_df[city_df['Город'] == selected_city]

    if selected_mode != "Все":
        city_df = city_df[city_df['Режим работы'] == selected_mode]

    if selected_exp != "Все":
        city_df = city_df[city_df['Опыт'] == selected_exp]

    # Отображаем таблицу с отфильтрованными данными
    st.dataframe(city_df, use_container_width=True)

    # Основные метрики
    st.subheader("Основные метрики")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего вакансий", len(city_df))
    with col2:
        st.metric("Уникальных вакансий", city_df['Название'].nunique())
    with col3:
        min_salary = city_df['min_salary'].min()
        st.metric("Минимальная зарплата", f"{min_salary:.0f} ₽")
    with col4:
        max_salary = city_df['max_salary'].max()
        st.metric("Максимальная зарплата", f"{max_salary:.0f} ₽")

    st.markdown("""**Данные csv файлы и dataframe созданы после обработки и удаления пропущенных значений.**""")

    # Распределение по категориям
    st.subheader("Распределение по категориям")

    # Создаем одну вкладку (не нужно использовать with для одной вкладки)
    tab = st.tabs(["Топ-10 вакансий"])[0]  # Берем первый (и единственный) элемент

    with tab:
        # Выбор города для анализа
        selected_city = st.selectbox(
            "Выберите город для анализа вакансий",
            ["Все города"] + list(df['Город'].unique()),
            key='city_selector'
        )

        # Фильтрация данных по выбранному городу
        if selected_city == "Все города":
            city_data = df
            title_suffix = "во всех городах"
        else:
            city_data = df[df['Город'] == selected_city]
            title_suffix = f"в {selected_city}"

        # Топ-10 вакансий по количеству
        top_vacancies = city_data['Название'].value_counts().nlargest(10)

        # Гистограмма топ-10 вакансий
        fig = px.bar(top_vacancies,
                     orientation='h',
                     title=f'Топ-10 вакансий {title_suffix}',
                     labels={'value': 'Количество вакансий', 'index': 'Название вакансии'},
                     color=top_vacancies.values,
                     color_continuous_scale='Blues')

        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Дополнительно: таблица с топ-10 вакансий
        st.write("Детализация топ-10 вакансий:")
        st.dataframe(top_vacancies.reset_index().rename(
            columns={'index': 'Вакансия', 'Название': 'Количество'}),
            use_container_width=True,
            hide_index=True)

# Раздел EDA
elif section == "EDA":
    st.header("Анализ вакансий по городам 📊")

    # 1. Выбор города
    selected_city = st.selectbox(
        "Выберите город для анализа",
        options=df['Город'].unique(),
        key='city_selector'
    )

    # Фильтрация данных по выбранному городу
    city_data = df[df['Город'] == selected_city]

    # 2. Круговая диаграмма: распределение по опыту работы (теперь без кольца)
    st.subheader(f"Распределение вакансий по опыту работы в {selected_city}")

    # Группируем данные по опыту работы
    exp_distribution = city_data['Опыт'].value_counts().reset_index()
    exp_distribution.columns = ['Опыт', 'Количество']

    fig_pie = px.pie(
        exp_distribution,
        values='Количество',
        names='Опыт',
        color='Опыт',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0,  # Теперь обычная круговая диаграмма (не кольцо)
        title=f'Доля вакансий по требуемому опыту в {selected_city}'
    )
    fig_pie.update_traces(
        textposition='outside',
        textinfo='percent+label',
        pull=[0.1, 0, 0, 0],  # Выделяем первый сегмент
        marker=dict(line=dict(color='#000000', width=0.5))
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # 3. Гистограмма: категории и опыт работы
    st.subheader(f"Распределение вакансий по категориям и опыту в {selected_city}")

    # Группируем данные
    category_exp_counts = city_data.groupby(['Категория', 'Опыт']).size().reset_index(name='Количество')

    fig_bar = px.bar(
        category_exp_counts,
        x='Категория',
        y='Количество',
        color='Опыт',
        barmode='group',
        title=f'Количество вакансий по категориям и опыту в {selected_city}',
        color_discrete_sequence=px.colors.qualitative.Set2,
        height=500
    )
    fig_bar.update_layout(
        xaxis_title="Категория вакансий",
        yaxis_title="Количество вакансий",
        legend_title="Требуемый опыт"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # 4. Диаграмма рассеяния: категории и режим работы (как в вашем примере)
    st.subheader(f"Распределение категорий вакансий по режиму работы в {selected_city}")

    # Добавляем небольшой случайный сдвиг (jitter) для лучшей визуализации точек
    np.random.seed(42)  # Для воспроизводимости
    city_data['jitter'] = np.random.uniform(-0.2, 0.2, size=len(city_data))

    fig_scatter = px.scatter(
        city_data,
        x='Категория',
        y='Режим работы',
        color='Категория',  # Цвет по категориям (можно заменить на 'Режим работы')
        hover_data=['Название', 'min_salary', 'max_salary'],
        title=f'Категории вакансий по режиму работы в {selected_city}',
        height=600,
        width=1000,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    # Настройка отображения
    fig_scatter.update_traces(
        marker=dict(size=12, opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Категория: %{x}<br>"
            "Режим работы: %{y}<br>"
            "Зарплата: %{customdata[1]:,.0f} - %{customdata[2]:,.0f} ₽<br>"
            "<extra></extra>"
        )
    )

    fig_scatter.update_layout(
        xaxis_title="Категория вакансий",
        yaxis_title="Режим работы",
        xaxis={'categoryorder': 'total descending'},  # Сортировка категорий по количеству
        showlegend=False  # Убираем легенду (если нужно, можно оставить)
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # 5. Дополнительная информация
    st.subheader(f"Ключевые метрики в {selected_city}")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Всего вакансий", len(city_data))

    with col2:
        st.metric("Уникальных категорий", city_data['Категория'].nunique())

    with col3:
        avg_salary = (city_data['min_salary'].mean() + city_data['max_salary'].mean()) / 2
        st.metric("Средняя зарплата", f"{avg_salary:,.0f} ₽")

    st.markdown(""" ### Характеристики 
**title**: Название - это название каждой вакансии, тип данных “object”  
**salary**: Зарплата - соответствующая каждой вакансии заработная плата, тип данных “object”  
**experience**: Опыт - необходимый опыт, который требуется для данной вакансии, тип данных “object”  
**work_format**: Режим работы - режим работы для каждой вакансии, тип данных “object”  
**Категория**: В этой колонке мы выделяем категорию для каждой вакансии по ключевым словам, тип данных “object”  
**min_salary**: Минимальная зарплата, тип данных “float”  
**max_salary**: Максимальная зарплата, тип данных “float”  
**Город**: Название города, с которого собраны вакансии, тип данных “object”  """)

# Раздел трендов
elif section == "Тренды":
    st.header("Тренды и закономерности 📈")

    # Создаем столбец со средней зарплатой (если его нет)
    df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2

    # Фильтры
    st.subheader("Фильтры")
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_cities = st.multiselect(
            "Выберите города",
            options=df['Город'].unique(),
            default=df['Город'].unique()
        )

    with col2:
        salary_range = st.slider(
            "Диапазон зарплат (₽)",
            min_value=int(df['min_salary'].min()),
            max_value=int(df['max_salary'].max()),
            value=(int(df['min_salary'].quantile(0.25)), int(df['max_salary'].quantile(0.75))))

    with col3:
        selected_categories = st.multiselect(
            "Выберите категории",
            options=df['Категория'].unique(),
            default=df['Категория'].unique()
        )

    # Применяем фильтры
    filtered_df = df[
        (df['Город'].isin(selected_cities)) &
        (df['avg_salary'].between(salary_range[0], salary_range[1])) &
        (df['Категория'].isin(selected_categories))
        ]

    # Визуализации - каждая в своем контейнере с выравниванием
    st.subheader("Диапазон зарплат по категориям")
    col1, col2, col3 = st.columns([1, 6, 1])  # Центральная колонка шире
    with col2:
        fig = px.box(
            filtered_df,
            x='Категория',
            y=['min_salary', 'max_salary'],
            title='Разброс зарплат по категориям',
            labels={'value': 'Зарплата (₽)', 'variable': 'Тип зарплаты'}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Сравнение минимальных и максимальных зарплат")
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        fig = px.scatter(
            filtered_df,
            x='min_salary',
            y='max_salary',
            color='Город',
            hover_name='Категория',
            title='Соотношение минимальных и максимальных зарплат'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Топ-5 самых высокооплачиваемых категорий")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**По минимальной зарплате:**")
        top_min = filtered_df.groupby('Категория')['min_salary'].max().nlargest(5)
        st.dataframe(top_min.reset_index().rename(columns={'min_salary': 'Мин. зарплата (₽)'}))

    with col2:
        st.markdown("**По максимальной зарплате:**")
        top_max = filtered_df.groupby('Категория')['max_salary'].max().nlargest(5)
        st.dataframe(top_max.reset_index().rename(columns={'max_salary': 'Макс. зарплата (₽)'}))

# Раздел выводов
elif section == "Выводы":
    st.header("Выводы")
    st.markdown("""
    1. Зарплаты: Санкт-Петербург — лидер по уровню зарплат, средняя зарплата выше, чем в Тольятти и Челябинске.
    2. Спрос: Наибольшее количество вакансий наблюдается в сферах менеджмента, продаж, сфере обслуживания и промышленной сфере (в Челябинске и Тольятти)
    3. Опыт работы: Большинство вакансий требуют минимального опыта работы от 1 до 3 лет или не требуют опыта работы совсем.
    4. Режим работы: В Санкт-Петербурге и остальных городах больше вакансий с “очным” форматом работы (т.е. на месте работодателя).
    """)

    st.subheader("Рекомендации")
    st.markdown("""
    - Соискателям: переезд в Санкт-Петербург может дать прирост зарплаты, но стоит учитывать стоимость жизни. В Тольятти и Челябинске меньше предложений с высокими зарплатами, но есть редкие исключения.
    - Работодателям: в Санкт-Петербурге конкуренция за кадры выше, нужно предлагать более высокие зарплаты. В Тольятти и Челябинске можно найти специалистов дешевле, но топ-кадры всё равно будут дорогими.
    """)

    st.subheader("Дальнейшие шаги")
    st.markdown("""
    Настроить парсинг данных по каждой вакансии, чтобы была возможность извлекать все требования по каждой вакансии, переходить по каждой и парсить соответствующую информацию: график работы, частота выплат и т.д.""")

