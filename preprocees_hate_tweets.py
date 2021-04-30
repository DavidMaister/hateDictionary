import pandas as pd
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# with this file we take the tweets from the database of hate_labeled_data.csv and put in a file, with each line a tweet
data = pd.read_csv('hate_labeled_data.csv')
data = data.rename(columns={'count': 'counter', 'class': 'type'})   # change names so they dont interfere with class names
#data = data.sort_values(by=['counter', 'offensive_language'])

# divide the dataset into which category belongs
# we are going to make two tests, one with offensive language and one with hate_speech

# 1. offensive language
data1_offensive = data[data.offensive_language/data.counter > data.neither/data.counter]
data1_neither = data[data.neither/data.counter > data.offensive_language/data.counter]
print('Offensive twits', data1_offensive.shape)
print('Neither twits', data1_neither.shape)
# as we have 20000 and 4282 twits, we are limited by the lowest number, and we want the data to be balanced
# we will take the 5000 most offensive language twits to make the dataset
data1_offensive = data1_offensive.sort_values(by=['offensive_language', 'counter'], ascending=[False, False])
data1_offensive = data1_offensive[:5000]
# concatenate both dataframes into a final one
data_offensive = data1_offensive.append(data1_neither)
# we shuffle it so the order is random and twits are mixed
data_offensive = shuffle(data_offensive)

# divide into train/test sets
train_offensive, test_offensive = train_test_split(data_offensive, test_size=0.2)

# create or erase content in case it is exists
f = open('offensive_twits_train.txt', 'w+')
# f.close()
# f = open('hate_twits.txt', 'a')  # append to the end of the file
for twit in train_offensive['tweet']:
    twit = re.sub(r"\bRT\b", "", twit)   # remove word RT
    twit = re.sub(r"[@#]\w+", "", twit)  # remove words starting with @ or #
    twit = re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", "", twit)     # remove links
    # https://code.tutsplus.com/tutorials/8-regular-expressions-you-should-know--net-6149 some regex explained

    f.write(twit + '\n')
f.close()



