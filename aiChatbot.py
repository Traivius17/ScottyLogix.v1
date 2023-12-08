import streamlit as st

import chatbot
import pandas as pd
import matplotlib.pyplot as plt


@st.cache_data
def chat(message):
    ints = chatbot.predict_class(message)
    response = chatbot.get_response(ints, chatbot.intents)
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
            ints = chatbot.predict_class(message)
            response = chatbot.get_response(ints, chatbot.intents)

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
