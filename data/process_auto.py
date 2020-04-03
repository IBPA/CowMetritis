import pandas as pd

column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin', 'name']
pd_data = pd.read_csv('./auto-mpg.data', delim_whitespace=True, names=column_names, encoding="utf-8-sig", na_values=['?'])
pd_data['ID'] = pd_data.index

# split data
bins = pd_data.mpg.quantile([.50])
low_mpgData = pd_data.loc[pd_data.mpg <= bins.iloc[0]]
high_mpgData = pd_data.loc[pd_data.mpg > bins.iloc[0]]

# copy data for future manipulation
low_mpgData_color = low_mpgData.copy()
high_mpgData_color = high_mpgData.copy()

# add column 'color' for each category
low_mpgData_color['Label'] = 'low'
high_mpgData_color['Label'] = 'high'

# concat dataset
dataset_concat = pd.concat([low_mpgData_color, high_mpgData_color])

dataset_concat.to_csv('./auto-mpg.csv', na_rep='.', index=False)
