import argparse
import gensim.downloader as api
from tqdm import tqdm
import io
import json
import numpy as np
import  random

from socialsent import seeds
from socialsent.representations.representation_factory import create_representation
from socialsent.polarity_induction_methods import random_walk, densify, pmi
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import time

parser = argparse.ArgumentParser()
parser.add_argument("corpus", help="set of words to analyse")
parser.add_argument("file_name", help="prefix of the file name that it is going to be generated")
# parser.add_argument("pos_seeds", help="set of positive seeds")
# parser.add_argument("neg_seeds", help="set of negative seeds")
# parser.add_argument("-e", "--embedding", help="embedding to use for the representation of each word ",
#                     default="glove-wiki-gigaword-50")
# parser.add_argument("-a", "--algorithm", help="algorithm to create the dictionary", default="socialsent")
#
args = parser.parse_args()

with io.open(args.corpus, 'r', encoding='utf8') as corpus:
    lines = corpus.readlines()
    random.shuffle(lines)   # shuffle so there is no order
word_set = set()  # using a set so there are no repeated words
for line in tqdm(lines):
    line_tokens = word_tokenize(line)
    # remove stopwords and punctuation tokens. it is a double condition
    stop_words = set(stopwords.words())
    tokens_without_sw = [word.lower() for word in line_tokens if word not in stop_words
                         if word not in string.punctuation]
    word_set.update(tokens_without_sw)
    # if aux == 20:
    #     break
    # aux += 1
    #if word_set.__len__() > 20000:   # 5000 ~ 5 minutos. 20000 is quite heavy
    #    break     # 8000 reviews break my memory
word_list = list(word_set)

model = api.load('glove-wiki-gigaword-50')
# seeds have to exist in the model if densify is used, if they are not in the model an error occurs
# if using densify Keras backend set to Theano
pos_seeds, neg_seeds = seeds.hist_seeds()
#neg_seeds = ["nigga", "bitch", "faggot", "nigger", "asshole", "motherfucker", "redneck", "wetback", "retard", "gipsy"]

print('Creating representations')
embeddings = create_representation("GIGA_fast", model, word_list + pos_seeds + neg_seeds)
#embedding_explicit = create_representation("Explicit", args.corpus)

print('Generating socialsent and densify dictionary')
tic = time.time()
polarities_socialsent = random_walk(embeddings, pos_seeds, neg_seeds, beta=0.99, nn=10, sym=True, arccos=True)
toc = time.time()
print('Time socialsent algorithm: ', toc-tic)
polarities_densify = densify(embeddings, pos_seeds, neg_seeds, beta=0.99, nn=10, sym=True, arccos=True)
tac = time.time()
print('Time densify algorithm: ', tac-toc)
# print('Generating pmi')
#polarities_pmi = pmi(embedding_explicit, pos_seeds, neg_seeds)
# polarities_socialsent =  dict(polarities_socialsent)
# polarities_densify =  dict(polarities_densify)
# values of polarities are float32 and they are needed to be float64 to be serializable by json

polarities_socialsent = dict([key, np.float64(value)] for key, value in polarities_socialsent.items())
polarities_densify = dict([key, np.float64(value)] for key, value in polarities_densify.items())

sorted_keys_socialsent = sorted(polarities_socialsent, key=polarities_socialsent.get)
sorted_keys_densify= sorted(polarities_densify, key=polarities_densify.get)
with open('dictionaries/' + args.file_name + '_socialsent.json', 'w') as f:
    json.dump(polarities_socialsent, f)
    print('Socialsent written')
with open('dictionaries/' + args.file_name + '_densify.json', 'w') as f2:
    json.dump(polarities_densify, f2)
    print('Densify written')

print('\n')