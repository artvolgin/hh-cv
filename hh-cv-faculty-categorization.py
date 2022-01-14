# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 19:26:21 2021

@author: Artem
"""

# =============================================================================
# 0. Libraries
# =============================================================================

import pandas as pd
import numpy as np
import os
import getpass
import requests
import re
import urllib
import time
from bs4 import BeautifulSoup


# =============================================================================
# 1. Data Loading
# =============================================================================

# Read the data
os.chdir("/data")
df_cv_edu = pd.read_pickle("df_cv_edu.obj")

# =============================================================================
# 2. Preprocessing
# =============================================================================

# Select only records with end date from 2010 to 2020
df_cv_edu = df_cv_edu[(df_cv_edu['end_date'] >= 2000) & (df_cv_edu['end_date'] <= 2020)]

# =============================================================================
# 3. Deduplication of faculty names
# =============================================================================

### 3.1 Preprocessing of the dataset names

# Remove CVs with missing faculty name
df_cv_edu = df_cv_edu[df_cv_edu['faculty_name'].notna()]

# Remove abreviatures
df_cv_edu = df_cv_edu[df_cv_edu['faculty_name'].apply(lambda name: not (name.isupper()))]

# Text preprocessing
df_cv_edu['faculty_name'] = df_cv_edu['faculty_name'].apply(
    lambda t: re.sub(r'[^A-Za-zА-Яа-я ]+', '', t.lower()).strip())
df_cv_edu = df_cv_edu[df_cv_edu['faculty_name'] != '']

# Tokenization and lemmatization
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
import pymorphy2
stopwords = get_stop_words('ru')
stopwords = stopwords + ["факультет", "институт", "наука", "школа", "высокий",
                         "академия", "faculty", "school", "бакалавр", "базовый",
                         "аспирантура", "асперантура", "аспирант", "бакалавриат",
                         "обучение", "заочный", "вечерний", "вечернезаочный",
                         "наука", "департмент", "дистанционный", "дополнительный",
                         "образование", "профессиональный", "открытый",
                         "кафедра", "магистерский", "программа", "подготовка",
                         "центр", "московский", "общий", "отдел", "ординатура",
                         "доктанатура", "открытый", "очнозаочный", "форма",
                         "последипломный", "постдипломный", "президентский",
                         "российский", "федерация", "приволжский", "межрегиональный",
                         "переподготовка", "повышение", "квалификация", "мва",
                         "ранхигс", "университет", "удостоверение", "удмуртский",
                         "университетский", "колледж", "уральский", "профиль",
                         "сотрудник"]
morph = pymorphy2.MorphAnalyzer()


def tokenizer_lemmatizer(string, stopwords=stopwords):
    '''
    Tokenize string and lemmatize words, remove stopwords
    '''
    
    string = string.replace('-', ' ')
    string = string.replace('—', ' ')
    result = [morph.parse(w)[0].normal_form for w in word_tokenize(string)]
    result = [w for w in result if w not in stopwords]
    
    return [morph.parse(w)[0].normal_form for w in word_tokenize(string) if w not in stopwords]


df_cv_edu['faculty_name_tokens'] = df_cv_edu['faculty_name'].apply(tokenizer_lemmatizer)
df_cv_edu['faculty_name_string'] = df_cv_edu['faculty_name_tokens'].apply(lambda t: " ".join(t))
df_cv_edu = df_cv_edu[df_cv_edu['faculty_name_string'] != '']


### 3.2 Preprocessing of faculty names

# Calculate freq of faculty names
freq_facultyname = df_cv_edu['faculty_name_string'].value_counts()
freq_facultyname = pd.DataFrame(freq_facultyname).reset_index()
freq_facultyname.columns = ['faculty_name', 'freq']
# Remove rare names
freq_facultyname = freq_facultyname[freq_facultyname['freq'] > 1]
# Remove short names
freq_facultyname['len'] = freq_facultyname['faculty_name'].apply(len)
freq_facultyname = freq_facultyname[freq_facultyname['len'] > 4]

# 1. Математические и естественные науки
# физич
r_1 = re.compile(r'математи | физик |  химия | химичес | статисти | геолог | географ | биолог |'
                 ' гидро | эколог | физмат | кибернет | земля | недропольз | естествен | природный', flags=re.I | re.X)
freq_facultyname['group_1'] = freq_facultyname['faculty_name'].apply(lambda name: int(bool(r_1.findall(name))))

# 2. Инженерное дело, технологии и технические науки
r_2 = re.compile((r'технолог | техничес | инженер | строител | информа | транспорт | компьютер | энергети | программир | бурен | оптика | энерго | вычислите | трактор | материал | прибор |'
                 ' производ | промышлен | автомат | связь | механик | механич | эксплуат | электро | техник | машин | газос | минерал | механиз | металл |'
                 ' робот | нефтян | нефть | газовый | аэро | космиче | двигател | ядерны | архитек | автомобил | дорожн | горный | авиаци | аппарат'), flags=re.I | re.X)
freq_facultyname['group_2'] = freq_facultyname['faculty_name'].apply(lambda name: int(bool(r_2.findall(name))))

# 3. Здравоохранение и медицинские науки
r_3 = re.compile((r'медицин | медико | сестринск | здравоохран | фармацев | лечеб | стоматоло | фармация'), flags=re.I | re.X)
freq_facultyname['group_3'] = freq_facultyname['faculty_name'].apply(lambda name: int(bool(r_3.findall(name))))

# 4. Сельское хозяйство и сельскохозяйственные науки
r_4 = re.compile((r'агроном | садовод | лесное | зоотехн | лесной | лесозаготов | дерево | сельское | почвовед | природо | овощевод | ветеринар | растительный | водоснабжение'), flags=re.I | re.X)
freq_facultyname['group_4'] = freq_facultyname['faculty_name'].apply(lambda name: int(bool(r_4.findall(name))))

# 5. Науки об обществе
r_5 = re.compile((r'социол | социал | бизнес | эконом | финанс | политолог | психолог | менеджмен | бухгалте | товаровед | политич | урбанис |'
                 ' торгов | регионовед | отношени | политик | реклам | обществен | журнал | маркет | логист | банков | демограф | менеджемент |'
                 ' туризм | сервис | гостини | юридич | юриспру | юрист | право | медиа | аудит | таможен | коммерц | налог | судебный | муниципальный'), flags=re.I | re.X)
freq_facultyname['group_5'] = freq_facultyname['faculty_name'].apply(lambda name: int(bool(r_5.findall(name))))

# 6. Образование и педагогические науки
r_6 = re.compile((r'педагог | детство | дефектолог | логопед | педиатр'), flags=re.I | re.X)
freq_facultyname['group_6'] = freq_facultyname['faculty_name'].apply(lambda name: int(bool(r_6.findall(name))))

# 7. Гуманитарные науки
r_7 = re.compile((r'археолог | философ | региовед | теолог | лингвист | истори | филолог |'
                  'гуманитар | язык | перевод | африка | восток | архив | физический культура'), flags=re.I | re.X)
freq_facultyname['group_7'] = freq_facultyname['faculty_name'].apply(lambda name: int(bool(r_7.findall(name))))

# 8. Искусство и культура
r_8 = re.compile((r'дизайн | искусств | хореограф | музыка | художествен'), flags=re.I | re.X)
freq_facultyname['group_8'] = freq_facultyname['faculty_name'].apply(lambda name: int(bool(r_8.findall(name))))

# Fix intersection between Math (Science) and Engineering
freq_facultyname['group_2'] = ((freq_facultyname.loc[:,['group_1', 'group_2']].sum(1) != 2) & (freq_facultyname.loc[:,['group_2']].sum(1) == 1)).astype(int)

# Fix intersection between Social Science and Humanities
freq_facultyname['group_5'] = ((freq_facultyname.loc[:,['group_5', 'group_7']].sum(1) != 2) & (freq_facultyname.loc[:,['group_5']].sum(1) == 1)).astype(int)

# Fix intersection between Social Science and Engineering
freq_facultyname['group_2'] = ((freq_facultyname.loc[:,['group_2', 'group_5']].sum(1) != 2) & (freq_facultyname.loc[:,['group_2']].sum(1) == 1)).astype(int)

# Fix intersection between Social Science and Education
freq_facultyname['group_5'] = ((freq_facultyname.loc[:,['group_5', 'group_6']].sum(1) != 2) & (freq_facultyname.loc[:,['group_5']].sum(1) == 1)).astype(int)

# Fix intersection between Art and Humanities
freq_facultyname['group_8'] = ((freq_facultyname.loc[:,['group_7', 'group_8']].sum(1) != 2) & (freq_facultyname.loc[:,['group_8']].sum(1) == 1)).astype(int)

# Fix intersection between Medicine and Agriculture
freq_facultyname['group_3'] = ((freq_facultyname.loc[:,['group_3', 'group_4']].sum(1) != 2) & (freq_facultyname.loc[:,['group_3']].sum(1) == 1)).astype(int)

# Fix intersection between Humanities and Engineering
freq_facultyname['group_2'] = ((freq_facultyname.loc[:,['group_2', 'group_7']].sum(1) != 2) & (freq_facultyname.loc[:,['group_2']].sum(1) == 1)).astype(int)


# Number of groups for each phrase
# freq_facultyname = freq_facultyname.drop('group_num', 1)
freq_facultyname['group_num'] = freq_facultyname.iloc[:,list(map(lambda s: s.startswith('group'), freq_facultyname.columns))].sum(1)

# Save for coding
freq_facultyname.rename(columns={'group_1':"Математические и естественные науки",
                                 'group_2':"Инженерное дело, технологии и технические науки",
                                 'group_3':"Здравоохранение и медицинские науки",
                                 'group_4':"Сельское хозяйство и сельскохозяйственные науки",
                                 'group_5':"Науки об обществе",
                                 'group_6':"Образование и педагогические науки",
                                 'group_7':"Гуманитарные науки",
                                 'group_8':"Искусство и культура"}, inplace=True)
freq_facultyname['group_num'].value_counts()

# freq_facultyname.to_excel("freq_facultyname.xlsx")


### 3.3 Add faculty category to the dataset
freq_facultyname = pd.read_excel("freq_facultyname_coded.xlsx")
faculty_name = freq_facultyname['faculty_name']
faculty_category = freq_facultyname.drop(columns=['faculty_name', 'freq', 'len', 'group_num'])
faculty_category = faculty_category.replace(1, pd.Series(freq_facultyname.columns, freq_facultyname.columns))
faculty_category = faculty_category.apply(lambda row: row[row != 0].values, 1)
faculty_category = faculty_category.apply(lambda row: row[0] if len(row) != 0 else np.nan)
dict_name_category = dict(zip(list(faculty_name),
                              list(faculty_category)))

df_cv_edu['faculty_category'] = df_cv_edu['faculty_name_string'].apply(
    lambda name: dict_name_category[name] if name in dict_name_category else np.nan)

### 3.4 Save to pickle
df_cv_edu = df_cv_edu.drop(columns=["faculty_name_tokens", "faculty_name_string"])
df_cv_edu.to_pickle("df_cv_edu_postprocess.obj")

