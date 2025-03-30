import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the data(stackoverflow_full.csv)

df=pd.read_csv('data/stackoverflow_full.csv')
df = df.head(5000)  # Keep only the first 1000 rows
df = df.drop(columns=['MentalHealth', 'MainBranch', 'YearsCode', 'YearsCodePro','Unnamed: 0','PreviousSalary','ComputerSkills','Accessibility'])
df = df.dropna(subset=['HaveWorkedWith'])
# Renaming the 'Employment' column to 'PrevEmploymentStatus'
df = df.rename(columns={'Employment': 'PrevEmploymentStatus'})


print(df.info())
print(df.isnull().sum())
print(df.describe(include='all'))

df.to_csv('data/cleaned_data.csv', index=False)

#visualizations
sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")

plt.figure(figsize=(8,5))
sns.countplot(x="Age", data=df)
plt.title("Age Distribution")

plt.figure(figsize=(8,5))
sns.countplot(x="Age", hue="Gender", data=df)
plt.title("Age vs Gender")
plt.show()

