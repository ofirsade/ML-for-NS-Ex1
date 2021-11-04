# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Task 1 - Load Data

col_names = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45', 'lpsa']
file = "/Users/ofirsade/Desktop/UNI/Masters/Courses/למידה חישובית למדעי המוח/HW/prostate_data.csv"
df = pd.read_csv(file, usecols = col_names)

# Task 2 - Scatter Plot of age and lpsa

plt.scatter(df['age'], df['lpsa'], marker = 'o')
plt.xlabel('Age')
plt.ylabel('Lpsa')
plt.show()

# Task 3 - Box Whisker plota of lpsa for ages a<60, 60<b<70, 70<c
lim_df = df[['lpsa', 'age']]

age_a_df = lim_df.loc[lim_df['age'] < 60]
print("AGE A DF:\n", age_a_df)
figa, axa = plt.subplots()
axa.set_title('lpsa for ages < 60')
axa.boxplot(age_a_df['lpsa'])

age_b_df = lim_df.loc[(lim_df['age'] > 60) & (df['age'] < 70)]
figb, axb = plt.subplots()
axb.set_title('lpsa for ages between 60 and 70')
axb.boxplot(age_b_df['lpsa'])

age_c_df = lim_df.loc[lim_df['age'] > 70]
figc, axc = plt.subplots()
axc.set_title('lpsa for ages greater than 70')
axc.boxplot(age_c_df['lpsa'])

# Task 4 - Adding a new column to the dataframe

ages = df[['age']]
med_age = np.median(ages)
comp_col = np.where(df['age'] > med_age, True, False)
df['age is greater than the median age'] = comp_col
#print("DF:\n", df)

# Task 5 - Print the indices of the subjects of age > 70

ind = (df.loc[df['age'] > 70]).index
print("\n", list(ind))

# Task 6 - Print the average lpsa of the subjects of age > 70

avg_lpsa = np.mean(age_c_df['lpsa'])
print("\nAverage lpsa of the subjects of age > 70: ", avg_lpsa)

# Task 7 - Print the records for which age > 70 and svi = 1

tmp_df = df.loc[df['age'] > 70]
df1 = tmp_df.loc[tmp_df['svi'] == 1]
print("\nRecords for which age > 70 and svi == 1:\n", df1)





