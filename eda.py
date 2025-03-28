import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the data(stackoverflow_full.csv)

df=pd.read_csv('data/stackoverflow_full.csv')
df = df.drop(columns=['MentalHealth', 'MainBranch', 'YearsCode', 'YearsCodePro','Unnamed: 0','PreviousSalary','ComputerSkills'])
df = df.dropna(subset=['HaveWorkedWith'])

print(df.info())
print(df.isnull().sum())
print(df.describe(include='all'))


sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")

plt.figure(figsize=(8,5))
sns.countplot(x="Age", data=df)
plt.title("Age Distribution")

plt.figure(figsize=(8,5))
sns.countplot(x="Age", hue="Gender", data=df)
plt.title("Age vs Gender")
plt.show()

