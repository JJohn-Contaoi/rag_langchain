import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

def get_advanced_template():
    template1 = ChatPromptTemplate.from_template('''
    You are a helpful assistant programmed to generate questions based only on the prided {context}. 
    Think a step by step and create a quiz with the number of {input} questions provided by user.

    Your output should be shaped as follows: 
    1. An outer list that contains 5 inner lists.
    2. Each inner list represents a set of question and answers, and contains exactly 4 strings in this order:
    - The generated question.
    - The correct answer.
    - The first incorrect answer.
    - The second incorrect answer.

    Your output should mirror this structure:
    [
        ["What is the complexity of a binary search tree?", "O(1)", "O(n)", "O(log n)", "O(n^2)"]
        ...
    ]

    Don't add introduction, note or conclusion. 
    ''')
    return template1

def get_intermediate_template():
    template2 = ChatPromptTemplate.from_template('''
    You are an expert quiz maker for intermediate technical fields. Think a step by step and 
    create a quiz with {input} questions based only on the provided context: {context}.

    The quiz type should be Multiple-choice:
    The format of the quiz type:
    - Questions: <context></context>
        <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
        <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
        ....
    - Answers:
        <Answer1>: <a|b|c|d>
        <Answer2>: <a|b|c|d>
        ....
    Example:
        - 1. The ____ is complexity of a binary search tree.
            a. O(1)
            b. O(n)
            c. O(log n)
            d. O(n^2)
        - Answers:
            1. b
    ''')
    return template2

def get_beginner_template():
    template3 = ChatPromptTemplate.from_template('''
    You are an expert quiz maker for a user-friendly beginner fields. Think a step by step and 
    create a quiz with {input} questions based only on the provided context: {context}.

    The quiz type should be Multiple-choice:
    The format of the quiz type:
    - Questions: <context></context>
        <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
        <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
        ....
    - Answers:
        <Answer1>: <a|b|c|d>
        <Answer2>: <a|b|c|d>
        ....
    Example:
        - 1. O(n) is complexity of a B_____ S_____ T___.
            a. Depth First Search
            b. Binary Search Tree
            c. Breadth First Search
            d. none of the above
        - Answers:
            1. b
    ''')
    return template3