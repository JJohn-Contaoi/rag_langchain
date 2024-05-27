import streamlit as st
import random 
import ast

def string_to_list(s, retries=10):
    for _ in range(retries):
        try:
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            continue
    st.write("Oops: Failed to load after multiple attempts. Please press Generate again.")
    st.stop()

def get_randomized_options(options):
    if not options:
        return [], None
    correct_answers = options[0]
    random.shuffle(options)
    return options, correct_answers