#!/usr/bin/env python
# coding: utf-8

# # EDA українського датасету для задачі виявлення спаму
# 
# Мета цього ноутбука -- виконати початковий аналіз даних, зрозуміти їхню структуру, виявити характерні ознаки спам-повідомлень і підготувати дані до побудови baseline-моделей.
# 
# У межах цього ноутбука виконується:
# 1. завантаження та первинна перевірка датасету;
# 2. аналіз структури, пропусків і дублікатів;
# 3. аналіз балансу класів;
# 4. аналіз довжини повідомлень;
# 5. аналіз частотних слів;
# 6. аналіз характерних ознак спаму (посилання, капслок, спецсимволи);
# 7. поділ на train/test вибірки;
# 8. формулювання висновків.

# In[1]:


import re
from collections import Counter

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")
pd.set_option("display.max_colwidth", 120)


# ## 1. Завантаження даних
# 
# Використовується датасет `amiadesu/ukrainian-social-spam` з Kaggle. 

# In[2]:


dataset_dir = kagglehub.dataset_download("amiadesu/ukrainian-social-spam")
print("Dataset directory:", dataset_dir)


# In[3]:


from pathlib import Path

dataset_path = Path(dataset_dir)
csv_files = list(dataset_path.rglob("*.csv"))
print("Found CSV files:", csv_files)

if not csv_files:
    raise FileNotFoundError("CSV-файл не знайдено в директорії датасету.")

csv_path = csv_files[0]
print("Using file:", csv_path)

data = pd.read_csv(csv_path)
data.head()


# ## 2. Первинний огляд структури датасету
# 
# На цьому кроці потрібно перевірити:
# - розмір датасету;
# - назви колонок;
# - типи даних;
# - приклади записів;
# - структуру цільової змінної.

# In[4]:


print("Shape:", data.shape)
print("\nColumns:", list(data.columns))
print("\nПерші 5 рядків:")
display(data.head())


# In[5]:


print("Інформація про датасет:")
data.info()


# In[6]:


print("Кількість унікальних значень у кожній колонці:")
display(data.nunique())

print("\nУнікальні значення цільової змінної 'spam':")
display(data["spam"].value_counts(dropna=False))


# ## 3. Перевірка якості даних: пропуски і дублікати
# 
# Для задач текстової класифікації важливо перевірити:
# - чи є пропуски в текстах;
# - чи є повністю дубльовані записи;
# - чи потрібно очищення перед подальшим аналізом.

# In[7]:


print("Пропуски по колонках:")
display(data.isnull().sum())

print("\nКількість повних дублікатів:", data.duplicated().sum())


# In[8]:


# За потреби можна очистити дані від дублікатів і пропусків.
# Нижче показано базовий варіант, але спершу варто оцінити, чи не втратимо корисні приклади.

clean_data = data.dropna(subset=["text", "spam"]).drop_duplicates().copy()

print("Розмір початкового датасету:", data.shape)
print("Розмір після базового очищення:", clean_data.shape)


# Якщо дублікати або пропуски є, потрібно вказати:
# - скільки саме записів було видалено;
# - чому це було зроблено;
# - як це може вплинути на якість моделі.

# ## 4. Аналіз балансу класів
# 
# Для задачі фільтрації спаму потрібно зрозуміти, чи є дисбаланс класів. 
# Це впливатиме на вибір метрик і, можливо, на налаштування моделей.

# In[9]:


class_counts = clean_data["spam"].value_counts().sort_index()
class_share = clean_data["spam"].value_counts(normalize=True).sort_index() * 100

print("Кількість об'єктів у класах:")
display(class_counts)

print("\nЧастка об'єктів у класах, %:")
display(class_share.round(2))


# In[10]:


plt.figure(figsize=(8, 5))
ax = sns.countplot(data=clean_data, x="spam")
ax.set_title("Розподіл класів")
ax.set_xlabel("Клас (0 = not spam, 1 = spam)")
ax.set_ylabel("Кількість повідомлень")
plt.show()


# Після цього блоку робимо короткий висновок:
# - чи є дисбаланс класів;
# - чи є він критичним;
# - чи потрібно враховувати його під час навчання моделей.

# ## 5. Аналіз довжини повідомлень
# 
# Довжина тексту часто є корисною ознакою: спам-повідомлення можуть бути або значно коротшими, або навпаки містити розширені рекламні формулювання, заклики до дії та посилання.

# In[11]:


clean_data["length_chars"] = clean_data["text"].astype(str).str.len()
clean_data["length_words"] = clean_data["text"].astype(str).str.split().apply(len)

display(clean_data[["length_chars", "length_words"]].describe().T)


# In[12]:


plt.figure(figsize=(12, 5))
sns.histplot(data=clean_data, x="length_chars", hue="spam", bins=60, kde=True, log_scale=True)
plt.title("Розподіл довжини повідомлень у символах (log scale)")
plt.xlabel("Довжина повідомлення, символи")
plt.ylabel("Кількість")
plt.show()


# In[13]:


plt.figure(figsize=(10, 5))
sns.boxplot(data=clean_data, x="spam", y="length_words")
plt.title("Порівняння довжини повідомлень у словах за класами")
plt.xlabel("Клас (0 = not spam, 1 = spam)")
plt.ylabel("Кількість слів")
plt.show()


# In[14]:


length_summary = clean_data.groupby("spam")[["length_chars", "length_words"]].agg(["mean", "median"])
display(length_summary)


# Тут потрібно пояснити:
# - чи відрізняються spam і not spam за довжиною;
# - чому на гістограмі використано логарифмічну шкалу;
# - чи може довжина тексту бути додатковою ознакою для класифікації.

# ## 6. Аналіз частотних слів
# 
# На цьому кроці варто подивитися, які слова найчастіше зустрічаються:
# - у спамі;
# - у звичайних повідомленнях.
# 
# Це допоможе зрозуміти, чи справді класи відрізняються лексично, а також обґрунтувати подальше використання TF-IDF.

# In[15]:


def tokenize_simple(text: str):
    text = str(text).lower()
    return re.findall(r"\b[\w'’\-]+\b", text, flags=re.UNICODE)

spam_tokens = []
ham_tokens = []

for text, label in zip(clean_data["text"], clean_data["spam"]):
    tokens = tokenize_simple(text)
    if label == 1:
        spam_tokens.extend(tokens)
    else:
        ham_tokens.extend(tokens)

spam_counter = Counter(spam_tokens)
ham_counter = Counter(ham_tokens)

top_spam = pd.DataFrame(spam_counter.most_common(20), columns=["word", "count"])
top_ham = pd.DataFrame(ham_counter.most_common(20), columns=["word", "count"])

print("ТОП-20 слів у spam:")
display(top_spam)

print("ТОП-20 слів у not spam:")
display(top_ham)


# In[16]:


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(data=top_spam.head(15), x="count", y="word", ax=axes[0])
axes[0].set_title("ТОП слів у spam")

sns.barplot(data=top_ham.head(15), x="count", y="word", ax=axes[1])
axes[1].set_title("ТОП слів у not spam")

plt.tight_layout()
plt.show()


# За бажанням сюди можна додати:
# - список стоп-слів для української мови;
# - очищення від дуже частих службових слів;
# - окремий аналіз біграм або триграм.
# 
# Але навіть базовий список частотних слів уже дає корисне розуміння структури текстів.

# ## 7. Аналіз характерних ознак спаму
# 
# Крім самих слів, для спаму часто характерні додаткові ознаки:
# - наявність URL;
# - велика частка великих літер;
# - надлишок спеціальних символів;
# - повтори знаків оклику або інших символів.

# In[17]:


# clean_data["has_url"] = clean_data["text"].astype(str).str.contains(r"http[s]?://|www\.", regex=True, case=False)
clean_data["has_url"] = clean_data["text"].astype(str).str.contains(r"__URL__", regex=True, case=False)
# будь-які символи поза літерами/цифрами/пробілами, тобто пунктуація, emoji та знаки на кшталт !, ?, ., ,, :, ;, @, #, $, %, &, *, (, ), -, _, =, +
clean_data["has_special_chars"] = clean_data["text"].astype(str).str.contains(r"[^\w\s]", regex=True)

def upper_ratio(text: str) -> float:
    text = str(text)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    upper_letters = [c for c in letters if c.isupper()]
    return len(upper_letters) / len(letters)

clean_data["upper_ratio"] = clean_data["text"].astype(str).apply(upper_ratio)

feature_summary = clean_data.groupby("spam")[["has_url", "has_special_chars", "upper_ratio"]].mean()
display(feature_summary)


# In[18]:


feature_plot = feature_summary.reset_index().melt(id_vars="spam", var_name="feature", value_name="value")

plt.figure(figsize=(10, 5))
sns.barplot(data=feature_plot, x="feature", y="value", hue="spam")
plt.title("Порівняння додаткових ознак між класами")
plt.xlabel("Ознака")
plt.ylabel("Середнє значення / частка")
plt.show()


# Після цього блоку варто письмово відповісти:
# - чи частіше спам містить посилання;
# - чи має спам більшу частку великих літер;
# - чи є спецсимволи характерною ознакою спаму.

# ## 8. Формалізація задачі
# 
# На основі проаналізованих даних задачу можна формалізувати як задачу бінарної класифікації:
# - вхід: текст повідомлення;
# - вихід: клас `spam` або `not spam`.
# 
# Для оцінювання моделей доцільно використовувати:
# - precision;
# - recall;
# - F1-score;
# - ROC-AUC.
# 
# Особливо важливо контролювати precision і recall, оскільки в системах фільтрації спаму небажані як пропуски спаму, так і хибне блокування нормальних коментарів.

# ## 9. Поділ на train/test вибірки
# 
# Після завершення EDA можна переходити до побудови baseline-моделей. 
# На цьому етапі дані слід поділити на тренувальну та тестову вибірки із збереженням пропорцій класів.

# In[19]:


X = clean_data["text"]
y = clean_data["spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

print("\nРозподіл класів у train, %:")
display((y_train.value_counts(normalize=True) * 100).round(2))

print("\nРозподіл класів у test, %:")
display((y_test.value_counts(normalize=True) * 100).round(2))


# ## 10. Підсумкові висновки
# 1. Датасет містить достатню кількість повідомлень для навчання моделей машинного навчання?
# 2. Цільова змінна придатна для постановки задачі бінарної класифікації?
# 3. Потрібно зазначити, чи були виявлені пропуски та дублікати.
# 4. Потрібно зазначити, чи є дисбаланс класів і наскільки він критичний.
# 5. Потрібно описати відмінності між spam і not spam за:
#    - довжиною повідомлень;
#    - словниковим складом;
#    - наявністю URL;
#    - використанням великих літер і спецсимволів.
# 6. На основі цих спостережень потрібно зробити висновок, що для baseline-експериментів доцільно використати TF-IDF і класичні моделі, а на наступному етапі — порівняти їх із BERT-подібною моделлю для української мови.

# # Wordclouds

# In[20]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

spam_text = data[data['spam'] == 1]['text']
ham_text = data[data['spam'] == 0]['text']

spam_words = " ".join(spam_text.astype(str))
ham_words = " ".join(ham_text.astype(str))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
wc_spam = WordCloud(width=800, height=400, background_color='white').generate(spam_words)
plt.imshow(wc_spam)
plt.title("Spam")
plt.axis("off")

plt.subplot(1, 2, 2)
wc_ham = WordCloud(width=800, height=400, background_color='white').generate(ham_words)
plt.imshow(wc_ham)
plt.title("Not spam")
plt.axis("off")

plt.show()


# In[23]:


def filter_short_words(text):
    if not isinstance(text, str):
        return ''

    # прибираємо пунктуацію
    text = re.sub(r'[^\w\s]', '', text)

    # фільтр слів
    words = [word for word in text.split() if len(word) >= 2]

    return ' '.join(words)

clean_data['text'] = clean_data['text'].str.replace(r'\b__USER__\b', '', regex=True)
clean_text = clean_data['text']
# clean_data['filtered_text'] = clean_data['text'].apply(filter_short_words)

# print(clean_data['filtered_text'])


# In[ ]:




