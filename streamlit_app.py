import streamlit as st
import gensim.downloader as api
import spacy
from gensim.models import KeyedVectors

"""
# Welcome to app to test gensim model

There is text box here to enter a question, 2 markscheme questions, and 1 response.
"""

# @st.cache_data()
# def load_model():
#     model = api.load('word2vec-google-news-300')
#     return model

# model = load_model()

with st.spinner('Loading model into memory...'):
    model= KeyedVectors.load("model.bin")

def load_spacy_model():
    nlp = spacy.blank("en")

    # Create a spaCy Vocab with custom vectors
    vocab = nlp.vocab

    # Copy word vectors from Gensim model to spaCy Vocab. Sliced to 1/30th size to prevent colab from crashing
    for word, vector in zip(model.index_to_key[:100000], model.vectors[:100000]):
        vocab.set_vector(word, vector)  

    return nlp

nlp = load_spacy_model()

# Text box to enter question
question = st.text_input('Enter question here:')

# Text box to enter markscheme question 1
markscheme_1 = st.text_input('Enter markscheme question 1 here:')

# Text box to enter markscheme question 2
markscheme_2 = st.text_input('Enter markscheme question 2 here:')

# Text box to enter response
response = st.text_input('Enter response here:')

mark_scheme= [markscheme_1, markscheme_2]

# Button to run model
if st.button('Run model'):
    ans= nlp(response)
    max_score=-1
    for ms_ans in mark_scheme:
        ms_ans= nlp(ms_ans)
        this_similarity= ans.similarity(ms_ans) 
        max_score= max(this_similarity, max_score)
    st.write('The similarity score is: ', max_score)

