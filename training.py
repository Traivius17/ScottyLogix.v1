import random
import pickle
import json
import numpy as np


import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences

intents = json.loads(open("intents.json").read())

words = []
classType = []
categorized = []
newWords = []

ignore_characters = ["?", "!", ",", "."]


for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        categorized.append(((word_list), intent["tag"]))
        if intent["tag"] not in classType:
            classType.append(intent["tag"])

lemmatizer = WordNetLemmatizer()

for word in words:
    if word not in ignore_characters:
        lemmatizer.lemmatize(word)
        newWords.append(word)


newWords = sorted(set(newWords))

classType = sorted(set(classType))

pickle.dump(newWords, open("newWords.pk1", "wb"))
pickle.dump(classType, open("classes.pk1", "wb"))


training = []

emptyOut = [0] * len(classType)


for category in categorized:
    bag = []
    word_patterns = category[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in newWords:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(emptyOut)

    output_row[classType.index(category[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
bags = [item[0] for item in training]
output_rows = [item[1] for item in training]


train_x = list(np.array(bags))
train_y = list(np.array(output_rows))

# random.shuffle(training)

# training = np.array(training)
# train_x = list(training[:, 0])
# train_y = list(training[:, 1])
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
hist = model.fit(
    np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1
)

model.save("perfectedModel.h5", hist)
print("Im finished, code has been served")
