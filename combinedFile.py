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

# from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import SGD
# from keras.preprocessing.sequence import pad_sequences

intents = json.loads(open("intents.json").read())

""" words = []
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

#model.save("perfectedModel.h5", hist)
#print("Im finished, code has been served")
 """

# import random
# import json
# import pickle
# import numpy as np

# import nltk

# from nltk.stem import WordNetLemmatizer

# from tensorflow import keras
from keras.models import load_model

# from keras.preprocessing.sequence import pad_sequences

lemmatizer = WordNetLemmatizer()


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


import streamlit as st


import matplotlib.pyplot as plt


@st.cache_data
def chat(message):
    ints = predict_class(message)
    response = get_response(ints, intents)
    return response


appPage = st.sidebar.selectbox("Select Page", ["Home", "AI Chatbot", "About App"])
creator = st.sidebar.write("Creator: Traivius Chappell")


if appPage == "Home":
    Title = st.markdown(
        "<center><h1> Scottie's Transportation LLC 🚚 </h1></center>",
        unsafe_allow_html=True,
    )

    moto = st.markdown(
        "<center><h2>Delivering miles of smiles<h2><center>", unsafe_allow_html=True
    )
    image = st.image("name.png", use_column_width=True)
    intro = st.markdown(
        """<center> <h3> Learn about the company with our own AI chatbot.<h3><center>""",
        unsafe_allow_html=True,
    )
    info = st.markdown(
        """<center><h3>ScottyLogix is tailored to respond to anything about the company to find out more information<h3><center> """,
        unsafe_allow_html=True,
    )

    if st.button("About App"):
        appPage = "About App"


if appPage == "About App":
    text = """
# Welcome to Our Chatbot Application!

This application harnesses the power of a Neural Network model 🧠 to assist you.

The model has been trained using a rich dataset comprised of preprocessed JSON intents. These intents encompass various tags, user patterns, and static chatbot responses.

### Understanding the Training Data:
- **Tags (Category):** They provide context and categorization.
- **Patterns (User Text):** Represent diverse user queries or inputs.
- **Responses:** Static but informative responses to cater to user queries.

The data has been meticulously curated, providing specific insights about our company. It's designed to help answer queries from potential employees interested in joining our fleet.

### Trained on the following data:
- Greetings
- Goodbyes
- Age
- Name
- Established
- Driver Pay Rate
- Freight Type
- Home Time Policy
- Company Culture
- Advancement
- Sick And Time Off
- Expectations on ELD
- Safety Record
- Incidents

*Please use inputs that are based around these categories for the best performance*

### *How to locate and use the chatbot*:

Please make use of the menu in the top left corner and select -AI Chatbot-

Type in your question into the text box and press "Enter" 

 Watch ScottyLogix respond below

### Continuous Improvement:
We believe in evolving! More data and potential patterns will continuously enrich the model. This enhancement aims to elevate the user experience, ensuring comprehensive answers to any queries you might have.

Let's embark on this journey together! Feel free to explore and interact with our chatbot.

### Enjoy 🤖!
"""
    st.markdown(text, unsafe_allow_html=True)
if appPage == "AI Chatbot":
    tab1, tab2 = st.tabs(["Chat", "Example Inputs"])
    with tab2:
        st.markdown(
            """ 
                ## Chatbot not responding correctly?
                    
                ### Our apologies if this is happening
                
                ### ScottyLogix will become more robust as it is trained with more data in the future for a better user experience
                
                #### Here are some example input ideas that reflect the categorical data the model was trained on that will improve the chatbot's responses:
                
                ## Example User Inputs:

                ### Greeting category inputs: 
                hello, hey, Greetings, what's up, hey, how is it going, hello there

                ### Goodbye category inputs:
                seeya, cya, see you later, goodbye, have a great day, take it easy, cao, I am leaving, later bro

                ### Age category inputs:
                "how old are you?", age, what is your age, how old?, how long you been around

                ### Name category inputs:
                what is your name

                ### Established category inputs:
                how long has the buisness been runnning, how long have you had your LLC

                ### Driver Pay Rate category inputs:
                compensation for drivers, how much do you pay your drivers

                ### Home Time Policy category inputs:
                how often can I go home, whats the limit for home time

                ### Freight Type category inputs:
                what type of freight do you all haul, what freight do you carry

                ### Company Culture category inputs:
                is this a good place to work, culture? , should I work here

                ### Advancement category inputs:
                are there opportunities for advancement, are drivers the only job roles, can a driver change job positions

                ### Benefits category inputs:
                are there any benefits packages, do you offer medical benefits, what else do you offer other than money

                ### Benefits category inputs:
                are there any benefits packages, do you offer medical benefits, what else do you offer other than money

                ### Distance category inputs:
                how many miles will i get a week, average miles a driver gets, how far do I have to go for a load
                
                ### Sick, Time Off category inputs:

                sick time policy, how many days off can i put in, what if i get sick

                ### Communication category inputs:

                company communication, type of communication environment, methods of communication

                ### Electronic Logging category inputs:
                logging speciffications, do i have to log my own hours and distance?, how is the logging done

                ### Safety inputs category:
                do you all have a good safety record, How do you all enforce safety, is safety a priority
                
                ### Incidents inputs category:
                hat if i break down, how do you all handle break downs, who do i call if i need help, emergencies

                
                    

            """
        )
    with tab1:
        Title2 = st.markdown(
            "<center><h2> AI Assistance </h2></center>", unsafe_allow_html=True
        )
        notice1 = st.markdown(
            "<center><p> Please note that chatbot is an AI assistant for truck drivers looking to learn about information regarding to the company </p></center>",
            unsafe_allow_html=True,
        )

        notice2 = st.markdown(
            "<center><p> For the best experience, act as if you are a truck driver wanting to learn about Scottie's Transportation LLC </p></center>",
            unsafe_allow_html=True,
        )

        notice3 = st.markdown(
            "<center><p> Not sure what ask?  Responses not accurate?  </p></center>",
            unsafe_allow_html=True,
        )

        notice = st.markdown(
            "<center><b><p> Refer to the --Example Inputs-- tab above </p><b></center>",
            unsafe_allow_html=True,
        )

        with st.spinner(text="Loading a response"):
            message = st.text_input("Message ScottyLogix...", "Enter your text here")
            ints = predict_class(message)
            response = get_response(ints, intents)

            st.markdown(f"### ScottyLogix: {response}")

            intents = [item["intent"] for item in ints]
            probabilities = [item["probability"] for item in ints]

            tableInfo = st.markdown(
                """


            <center> <p> The table presented below provides a visual breakdown of how the model categorizes the provided inputs. Additionally, it showcases the probability estimation that the model assigns to each input category, indicating the perceived likelihood of your input falling into these specific categories.</p><center>

            <center> <p> Furthermore, based on the identified category, the model generates corresponding responses tailored to each category.</p><center>
                                    """,
                unsafe_allow_html=True,
            )
            # Create a bar chart using Matplotlib
            data = {"Category": intents, "Probability": probabilities}
            df = pd.DataFrame(data)
            table = st.table(df)
