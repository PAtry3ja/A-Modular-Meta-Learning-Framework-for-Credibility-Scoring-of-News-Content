import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('data/processed/cleaned_data.csv')

# 1. Rozkład klas
sns.countplot(data=df, x='label')
plt.title('Fake (0) vs Real (1)')
plt.show()

# 2. Długość tekstu
df['text_len'] = df['text'].apply(lambda x: len(str(x).split()))
sns.histplot(data=df, x='text_len', hue='label', bins=50, kde=True)
plt.title('Length of articles (Fake vs Real)')
plt.legend(title='Label', labels=['Fake', 'Real'])
plt.show()

# 3. Tematy vs klasy
pd.crosstab(df['subject'], df['label']).plot(kind='bar', stacked=True)
plt.title('Subjects of articles (Fake vs Real)')
plt.xlabel('Subject')
plt.ylabel('Amount of articles')
plt.legend(title='Label', labels=['Fake', 'Real'])
plt.tight_layout()
plt.show()

# 4. Liczba artykułów w czasie
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.groupby([df['date'].dt.year, 'label']).size().unstack().plot()
plt.title('Quantity of articles over time (by class)')
plt.legend(title='Label', labels=['Fake', 'Real'])
plt.xlabel('Year')
plt.ylabel('Amount')
plt.tight_layout()
plt.show()

