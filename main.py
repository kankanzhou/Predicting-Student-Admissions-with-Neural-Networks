import pandas as pd
import numpy as np

df = pd.read_csv('./data/student_data.csv')
df = pd.get_dummies(df, columns=['rank'])


df['gre'] = df['gre']/800
df['gpa'] = df['gpa']/4.0

sample = np.random.choice(df.index, size=int(len(df)*0.9), replace=False)
train_data, test_data = df.iloc[sample], df.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))
print(train_data[:10])
print(test_data[:10])


train_x = train_data.drop('admit', axis=1)
train_y = train_data['admit']
test_x = test_data.drop('admit', axis=1)
test_y = test_data['admit']

print(train_x[:10])
print(train_y[:10])


# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))
    
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)