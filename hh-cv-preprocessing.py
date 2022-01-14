# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 17:28:14 2020

@author: Artem
"""


# =============================================================================
# 0. Libraries
# =============================================================================

import pandas as pd
import numpy as np
import os
import getpass
import matplotlib.pyplot as plt
import math
# import pickle5 as pickle

# =============================================================================
# 1. Data Loading
# =============================================================================

# Read the data
os.chdir("/data")
df_cv = pd.read_csv("hh_cv_main.csv")
df_cv_edu = pd.read_pickle("df_cv_edu_postprocess.obj")
df_cv_edu = df_cv_edu.drop(columns='university_name')
df_cv_edu = df_cv_edu.rename(columns={'university_name_busgov':'university_name'})

# =============================================================================
# 2. Combine cv and cv_edu
# =============================================================================

# Keep only last obtained education
df_cv_edu = df_cv_edu.sort_values('end_date', ascending=False)
df_cv_edu = df_cv_edu.drop_duplicates("index1")

# Select only CVs with identifiable faculty category
# TODO: Reduce perecentage of NAs
df_cv_edu = df_cv_edu[df_cv_edu['faculty_category'].notna()]

# Merge education with main database
df_cv['index1'] = df_cv['index1'].astype(str)
df_cv = df_cv.merge(df_cv_edu)


# =============================================================================
# 3. Preprocessing
# =============================================================================

# ----- 3.1 Dates
# Select only CVs from 2010-2020
df_cv = df_cv[df_cv['year_of_cv_creation'] >= 2010]

# ----- 3.2 Skills
df_cv['skills_present'] = df_cv.skills_list.apply(lambda x: x != "[None]")

df_cv.groupby('year_of_cv_creation').size() * df_cv.groupby('year_of_cv_creation')['skills_present'].mean()

#  List of skills was initially saved as a string. Need to get np.array from string
def getEvalVariable(x):

    if x == '[None]':
        x = np.nan
    else:
        x = np.array(eval(x))
        
    return x


df_cv['skills_list'] = df_cv['skills_list'].apply(getEvalVariable)

# Skills frequency
skills_freq = df_cv['skills_list']
skills_freq = skills_freq[skills_freq.notna()]
skills_freq = pd.Series(np.concatenate(skills_freq.values)).value_counts()
skills_freq = pd.DataFrame(skills_freq).reset_index()
skills_freq.columns = ["skill", "freq"]
skill_to_category = pd.read_excel("skill_to_category.xlsx")
skill_to_category = skill_to_category[skill_to_category['category'].notna()]

'''
category_to_large = pd.read_excel("category_to_large.xlsx")
dict_category_large = dict(zip(list(category_to_large['category']),
                               list(category_to_large['large_category'])))
skill_to_category['category'] = skill_to_category['category'].apply(
    lambda x: dict_category_large[x])
'''
dict_skill_category = dict(zip(list(skill_to_category['skill']),
                               list(skill_to_category['category'])))
skills_freq['category'] = skills_freq['skill'].apply(
    lambda s: dict_skill_category[s] if s in dict_skill_category else np.nan)
category_freq = skills_freq.groupby('category')['freq'].sum()

# Skills categorization
skill_to_category = pd.read_excel("skill_to_category.xlsx")

keep_categories = {"Аналитические и исследовательские",
                   "Коммуникативные, выступления",
                   "Коммуникативные, командные",
                   "Коммуникативные, переговорные",                
                   "Компьютерные, программирование",          
                   "Компьютерные, специализированные программы",
                   "Написание и редактирование текстов",
                   "Медицинские, медико-психологические",
                   "Проведение тренингов и обучения",
                   "Работа с информацией в интернете (SMM, поиск, и т.д.)",
                   "Управление людьми", 
                   "Управление проектами",
                   "Финансовые",
                   "Юридические",
                   'Компьютерные программы, дизайн, архитектура и геодезия',
                   'Компьютерные программы, финансовые, административные и аналитические'}

skill_to_category = skill_to_category[skill_to_category['category'].apply(
    lambda category: category in keep_categories)]


dict_category_translate = {"Аналитические и исследовательские":"analytical",
                           "Иностранный язык":"foreign_language",
                           "Коммуникативные, выступления":"social_presentation",
                           "Коммуникативные, командные":"social_teamwork",
                           "Коммуникативные, общие":"social_general",
                           "Коммуникативные, переговорные":"social_negotiation",
                           "Компьютерные, программирование":"computer_programming",
                           "Компьютерные, специализированные программы":"computer_specsoftware",
                           "Написание и редактирование текстов":"writing",
                           "Финансовые":"financial",
                           "Юридические":"legal",
                           "Компьютерные, базовые":"computer_basic",
                           "Личностные качества":"personal",
                           "Медицинские, медико-психологические":"medical",
                           "Проведение тренингов и обучения":"training",
                           "Работа с информацией в интернете (SMM, поиск, и т.д.)":"smm",
                           "Управление людьми":"people_managment",
                           "Управление проектами":"project_managment",
                           'Компьютерные программы, дизайн, архитектура и геодезия':"computer_design_geo",
                           'Компьютерные программы, финансовые, административные и аналитические':"computer_financial_administrative"}
skill_to_category['category'] = skill_to_category['category'].replace(dict_category_translate)

# Create dummy for each skill category
for category in skill_to_category['category'].unique():
    
    selected_skills = set(skill_to_category[skill_to_category['category'] == category].skill)
    
    df_cv[('skills_' + category)] = df_cv['skills_list'].apply(
        lambda skills: len(set(skills).intersection(selected_skills)) != 0
        if type(skills) == np.ndarray else 0).astype(np.float64)
    print(category)

# ----- 3.3 Years of experience
# Remove outliers
df_cv = df_cv[df_cv.years_of_experience < 50]

# Split on the same intervals as in vacancies
df_cv['years_of_experience_interval'] = pd.cut(df_cv.years_of_experience,
                                               [0, 1, 3, 6, math.inf],
                                               right=False,
                                               labels=["0-1 year", "1-3 years", "3-6 years", "6+ years"])

# ----- 3.4 Expected salary

# Expected salary categorization: Nonzero, Zero, Missing
def getSalaryCategory(salary):
    if np.isnan(salary):
        result = "missing"
    elif salary == 0:
        result = "zero"
    else:
        result = "nonzero"
    return result


df_cv['expected_salary_category'] = df_cv['expected_salary'].apply(getSalaryCategory)


# Remove salary outliers
salary_quantile = df_cv['expected_salary'].quantile(0.995)
df_cv[df_cv['expected_salary'] >= salary_quantile]['expected_salary'] = salary_quantile


# ----- 3.5 Years after graduation

# Create the numeric variable
df_cv['years_after_graduation'] = df_cv['year_of_cv_creation'] - df_cv['end_date']

# Remove CVs with imposible years after graduation
df_cv = df_cv[df_cv['years_after_graduation'] >= -5]

# Currently student or already graduated
df_cv['student_status'] = (df_cv['years_after_graduation'] < 0).apply(lambda x: "student" if x else "graduate")

# ----- 3.6 Education level

# Combine doctor and candidate
df_cv["education_level"] = df_cv["education_level"].replace({"doctor":"candidate"})


# ----- 3.7 Age groups

# Remove CVs of 14-17 years of old applicants
df_cv = df_cv[df_cv['age_group'] != '14–17']

# Recode 55-64 and 65-100 to 55+
df_cv['age_group'] = df_cv['age_group'].replace({"55–64":"55+", "65–100":"55+"})


# ----- 3.8 Region of the university

# Remove CVs with missing regions
df_cv = df_cv[df_cv['region_name'].notna()]

# Create variable of university_region
table_uninames_hh_busgov = pd.read_excel("uninames_hh_busgov.xlsx")
dict_uni_region = dict(zip(table_uninames_hh_busgov['name_busgov'].values,
                           table_uninames_hh_busgov['region_busgov'].values))
df_cv['university_region_name'] = df_cv['university_name'].replace(dict_uni_region)

# Correct names
dict_correct_names = {
    'Еврейская автономная область':'Еврейская АО',
    'Республика Северная Осетия-Алания':'Республика Северная Осетия — Алания',
    'Кабардино-Балкарская республика':'Кабардино-Балкарская Республика',
    'Республика Карачаево-Черкесия':'Карачаево-Черкесская Республика',
    'Ханты-Мансийский автономный округ Югра':'Ханты-Мансийский АО - Югра'
}

df_cv['university_region_name'] = df_cv['university_region_name'].replace(dict_correct_names)
df_cv['region_name'] = df_cv['region_name'].replace(dict_correct_names)

# Applicant is looking for a job at the same region as the location of his/her university
df_cv['home_region'] = (df_cv['region_name'] == df_cv['university_region_name']).astype(int)


# ----- 3.9 Subsetting

# Select CVs after 2014
df_cv_skills = df_cv[df_cv['year_of_cv_creation'] >= 2014]

selected_columns = ["year_of_cv_creation", "region_name", 'university_region_name', 'home_region', "professional_area",
                    'gender', 'age_group', 'expected_salary', 'expected_salary_category',
                    'education_level', 'years_of_experience',
                    'university_name', 'end_date', 'faculty_category', 'years_of_experience_interval',
                    'years_after_graduation', 'student_status']
selected_columns = selected_columns + list(filter(lambda colname: colname.startswith("skills_"), df_cv_skills.columns))
selected_columns.remove("skills_list")
selected_columns.remove("skills_present")
df_cv_skills = df_cv_skills[selected_columns]

# ----- 3.10 Translation

# 1. Professional areas
dict_translation_professional_area = {
    "Маркетинг, реклама, PR":"Marketing, advertising, PR",
    "Строительство, недвижимость":"Construction, Real estate",
    "Продажи":"Sales",
    "Безопасность":"Security",
    "Административный персонал":"Administrative staff",
    "Бухгалтерия, управленческий учет, финансы предприятия":"Accounting, Finance",
    "Производство, сельское хозяйство":"Manufacturing, Agriculture",
    "Информационные технологии, интернет, телеком":"Information technology",
    "Банки, инвестиции, лизинг":"Banks, Investments, Leasing",
    "Медицина, фармацевтика":"Medicine, Pharmacy",
    "Высший менеджмент":"Top-management",
    "Начало карьеры, студенты":"Early career, Students",
    "Туризм, гостиницы, рестораны":"HoReCa",
    "Инсталляция и сервис":"Installation and Service",
    "Транспорт, логистика":"Transport, Logistics",
    "Управление персоналом, тренинги":"HR management, Training",
    "Искусство, развлечения, масс-медиа":"Art, Entertainment, Mass media",
    "Наука, образование":"Science, Education",
    "Юристы":"Lawyers",
    "Спортивные клубы, фитнес, салоны красоты":"Fitness, Beaty salons",
    "Государственная служба, некоммерческие организации":"Public service, Non-profit organizations",
    "Консультирование":"Consulting",
    "Добыча сырья":"Mining",
    "Страхование":"Insurance",
    "Автомобильный бизнес":"Car buisness",
    "Закупки":"Procurement",
    "Рабочий персонал":"Workers",
    "Домашний персонал":"Domestic servants"
    }

df_cv_skills['professional_area'] = df_cv_skills['professional_area'].replace(
    dict_translation_professional_area)

# 2. Region
from transliterate import translit, get_available_language_codes

dict_translation_region_name = df_cv_skills.region_name.unique()
dict_translation_region_name = list(map(lambda name: translit(name, reversed=True),
                                        dict_translation_region_name))
dict_translation_region_name = dict(zip(
    list(df_cv_skills.region_name.unique()),
    dict_translation_region_name))

df_cv_skills['region_name_rus'] = df_cv_skills['region_name']
df_cv_skills['region_name'] = df_cv_skills['region_name'].replace(dict_translation_region_name)
df_cv_skills['region_name'] = df_cv_skills['region_name'].replace({
    'Moskva':'Moscow',
    'Sankt-Peterburg':'Saint Petersburg'
    })
t = df_cv_skills['region_name_rus'].value_counts()

# 3. Faculties
# print(*df_cv_skills['faculty_category'].unique(), sep="\n")
dict_translation_faculty_category = {
    "Науки об обществе":"Social science",
    "Инженерное дело, технологии и технические науки":"Engineering and Technology",
    "Гуманитарные науки":"Humanities",
    "Здравоохранение и медицинские науки":"Health and Medicine",
    "Образование и педагогические науки":"Education",
    "Математические и естественные науки":"Mathematics and Natural science",
    "Сельское хозяйство и сельскохозяйственные науки":"Agricultural science",
    "Искусство и культура":"Art and Culture"
    }

df_cv_skills['faculty_category'] = df_cv_skills['faculty_category'].replace(
    dict_translation_faculty_category)

# ----- 3.11 Save to Excel
df_cv_skills.to_excel("df_cv_skills.xlsx")
df_cv_skills = pd.read_excel("df_cv_skills.xlsx")

