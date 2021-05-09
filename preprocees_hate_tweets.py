import pandas as pd
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# with this file we take the tweets from the database of hate_labeled_data.csv and put in a file, with each line a tweet
data = pd.read_csv('hate_labeled_data.csv')
data = data.rename(columns={'count': 'counter', 'class': 'type'})   # change names so they dont interfere with class names
data = data.sort_values(by='counter', ascending=False)


# divide the dataset into the category it belongs
# we are going to make two tests, one with offensive language and one with hate_speech

# separate datasets into neither, offensive and hate speech
data_hate = data[data.type == 0][['counter', 'type', 'tweet']]
data_offensive = data[data.type == 1][['counter', 'type', 'tweet']] #data[data.offensive_language/data.counter > data.neither/data.counter]
data_neither = data[data.type == 2][['counter', 'type', 'tweet']]

print('Hate twits', data_hate.shape)    # ->1430
print('Offensive twits', data_offensive.shape)  # -> 19190
print('Neither twits', data_neither.shape)  # -> 4163

# as we have dispair number of tweets for each class, and we want the data to be balanced
# we will make the two datasets with the minimum of the values as the limiting one
# concatenate both dataframes into a final one
dataset_offensive = data_offensive[0:5000].append(data_neither)
dataset_hate = data_hate.append(data_neither[0:2000])
# we shuffle it so the order is random and twits offensive and neither are mixed
dataset_offensive = shuffle(dataset_offensive)
dataset_hate = shuffle(dataset_hate)
# divide into train/test sets
train_offensive, test_offensive = train_test_split(dataset_offensive, test_size=0.2)
train_hate, test_hate = train_test_split(dataset_hate, test_size=0.2)

# save train files

# create or erase content in case it is exists
f = open('offensive_twits_train.txt', 'w+')
for twit in train_offensive['tweet']:
    twit = re.sub(r"\bRT\b", "", twit)   # remove word RT
    twit = re.sub(r"[@#]\w+", "", twit)  # remove words starting with @ or #
    twit = re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", "", twit)     # remove links
    # https://code.tutsplus.com/tutorials/8-regular-expressions-you-should-know--net-6149 some regex explained

    f.write(twit + '\n')
f.close()

f = open('hate_twits_train.txt', 'w+')
for twit in train_hate['tweet']:
    twit = re.sub(r"\bRT\b", "", twit)   # remove word RT
    twit = re.sub(r"[@#]\w+", "", twit)  # remove words starting with @ or #
    twit = re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", "", twit)     # remove links
    # https://code.tutsplus.com/tutorials/8-regular-expressions-you-should-know--net-6149 some regex explained

    f.write(twit + '\n')
f.close()

# save test files as csv
test_offensive.to_csv('offensive_twits_test.csv')
test_hate.to_csv('hate_twits_test.csv')




