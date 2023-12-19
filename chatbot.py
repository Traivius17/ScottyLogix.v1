import random
import json
import pickle
import numpy as np

import nltk
nltk.download("punkt")
nltk.download("wordnet")

from nltk.stem import WordNetLemmatizer

from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

with open("newWords.pk1", "rb") as words_file:
    words = pickle.load(words_file)

with open("classes.pk1", "rb") as classes_file:
    classes = pickle.load(classes_file)


model = load_model("perfectedModel.h5")


def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return bag


def predict_class(sentence):
    bow = bag_of_words(sentence)

    # new_shape = 187  # Specify the desired shape
    # bow = np.resize(bow, new_shape)

    # Check the shape of the bag-of-words representation
    # print("Bag of words shape:", bow.shape)

    res = model.predict(np.array([bow]))[0]
    ERROR_THRESOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list


# predict_class("what is your name")


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


# print("Bot is currrently running")

# while True:
#     message = input("Please enter some text: ")
#     ints = predict_class(message)
#     res = get_response(ints, intents)
#     print(res)
