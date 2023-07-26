import streamlit as st
import pandas as pd

df = pd.read_csv('Quiz.csv')
score = 0

with st.sidebar:
    st.write(f"CSV file Quiz created successfully.")
    with open('Quiz.csv', "rb") as f:
                st.download_button("Download CSV", f.read(), file_name="Quiz.csv", mime="application/csv")

# Streamlit app
# def main():
st.title("Quiz App")
st.write("Answer the following questions:")
# with st.form('quiz'):
for i,row in df.iterrows():
    # for i,j in zip(out.quiz,out.answer):
    st.write(f"\n{i+1}. {row['Question']}")
    # selected_option = st.radio("Select an option:", question_data['options'])
    selected_option = st.radio(f"Select an option##{i}", [row['OptionA'],row['OptionB'],row['OptionC'],row['OptionD']])

    # if selected_option == question_data['correct_option']:
    if selected_option == row['CorrectAnswer']:
        score += 1
    # submitQuiz = st.form_submit_button("Submit")

# print("heyyyy")
# st.write("\nYou've completed the quiz!")
# st.write(f"Your total score is: {score}/{df.shape[0]}")
    
if (st.button("Submit")):
    st.write("\nYou've completed the quiz!")
    st.write(f"Your total score is: {score}/{df.shape[0]}")
