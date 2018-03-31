import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# column names
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', 'target']

# Read in the CSV file
data = pd.read_csv('adult.test.csv', names=column_names, header=0)


# Remove the leading space in categorical features
cols = data.columns
num_cols = data._get_numeric_data().columns
cat_clos = list(set(cols) - set(num_cols))

# get rid of the space in the begining of categorical features
for cat_name in cat_clos:
    data[cat_name] = data[cat_name].apply(str.strip)


# remove the missing values
def missing_to_NA(data):
    if data == '?':
        result = np.nan
    else:
        result = data
    return result


def remove_missing(data_frame):
    cols = data_frame.columns
    for column_name in cols[:-1]:
        data[column_name] = data[column_name].apply(missing_to_NA)
    return data


data = remove_missing(data)

# drop na rows
clean_data = data.dropna()

# Convert target features into 0 and 1


def target_convert(word):
    if word == '<=50K.':
        result = 0
    else:
        result = 1
    return result


clean_data['target'] = clean_data['target'].apply(target_convert)

clean_data.to_csv('clean_test_data.csv', sep=',', encoding='utf-8', index=False)
