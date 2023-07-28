import os
import openai
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI,Cohere
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
import pandas as pd
import streamlit as st


st.title("QuizzyBrains")

with st.sidebar:
    with st.form('Cohere/OpenAI'):
        # User selects the model (OpenAI/Cohere) and enters API keys
        model = st.radio('Choose OpenAI/Cohere', ('OpenAI', 'Cohere'))
        api_key = st.text_input('Enter API key', type="password")

        submitted = st.form_submit_button("Submit")


# Check if API key is provided and set up the language model accordingly
mod = ""
if api_key:
    if model == 'OpenAI':
        os.environ["OPENAI_API_KEY"] = api_key
        model_name = "gpt-3.5-turbo-16k"
        temperature = 0.0
        model = OpenAI(model_name=model_name, temperature=temperature)
        mod = 'OpenAI'
    elif model == 'Cohere':
        os.environ["Cohere_API_KEY"] = api_key
        model = Cohere(cohere_api_key=api_key)
        mod = 'Cohere'


def quesAns(Topic,Level,NoOfQue):
    # Here's another example, but with a compound typed field.
    class Actor(BaseModel):
    #     name: str = Field(description="name of an actor")
    #     film_names: List[str] = Field(description="list of names of films they starred in")
        quiz: List[dict] = Field(description="question and list of 4 options of this question")
        answer: List[dict] = Field(description="Correct answer and a string of reason behind it of the question")
        

    actor_query = f"Generate the quiz of {NoOfQue} questions and options and correct answer for the same. Topic is {Topic} and Level is {Level}"

    parser = PydanticOutputParser(pydantic_object=Actor)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    _input = prompt.format_prompt(query=actor_query)

    output = model(_input.to_string())

    # parser.parse(output)
    out = parser.parse(output)

    return out

def tocsv(df):
    df.to_csv('pages/Quiz.csv',index=False)
    df.to_csv('Quiz.csv',index=False)

Topic_description = st.text_input('Enter Topic')
Level = st.selectbox("Choose the Level",("Easy","Medium","Hard"))
if(mod=='OpenAI'):
    numbers_list = [i for i in range(1, 26)]
    NoOfQue = st.selectbox("choose No of questions",numbers_list)
if(mod=='Cohere'):
    numbers_list = [i for i in range(1, 6)]
    NoOfQue = st.selectbox("choose No of questions",numbers_list)    

if (st.button('Submit')):
    st.write("Loading Questions...")
    out = quesAns(Topic_description, Level,NoOfQue)
    # st.write("question")
    ques = []
    optionA = []
    optionB = []
    optionC = []
    optionD = []
    CorrectAnswer = []
    Reason = []
    for i,j in zip(out.quiz,out.answer):
        ques.append(i['question'])
        optionA.append(i['options'][0])
        optionB.append(i['options'][1])
        optionC.append(i['options'][2])
        optionD.append(i['options'][3])
        CorrectAnswer.append(j['correct_answer'])
        Reason.append(j['reason'])

    df = pd.DataFrame(zip(ques,optionA,optionB,optionC,optionD,CorrectAnswer,Reason),columns =['Question', 'OptionA','OptionB','OptionC','OptionD','CorrectAnswer','Reason'])
    tocsv(df)
    st.markdown('<a href="/QA_test" target="_self">Go to Test -></a>', unsafe_allow_html=True)


