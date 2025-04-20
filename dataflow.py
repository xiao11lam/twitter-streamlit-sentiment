import re
from datetime import timedelta
from collections import defaultdict

from bytewax.dataflow import Dataflow
from bytewax.inputs import ManualInputConfig
from bytewax.outputs import StdOutputConfig, ManualOutputConfig
from bytewax.execution import run_main
from bytewax.window import TumblingWindowConfig, SystemClockConfig

import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from scipy.special import softmax
import pandas as pd
import json
import requests
from dotenv import load_dotenv
import os

from contractions import contractions

# load environment variables
load_dotenv()

# load spacy stop words
en = spacy.load('en_core_web_sm')
en.Defaults.stop_words |= {"s","t",}
sw_spacy = en.Defaults.stop_words

# load contractions and compile regex
pattern = re.compile(r'\b(?:{0})\b'.format('|'.join(contractions.keys())))

# load sentiment analysis model
MODEL = "model/"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# window size in minutes
WINDOW_SIZE = 1

# Instagram credentials
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")

# set up page details
st.set_page_config(
    page_title="Live Instagram Comments Sentiment Analysis",
    page_icon="📸",
    layout="wide",
)


def get_instagram_comments():
    """
    Get comments from Instagram
    """
    INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
    # Get recent media posts first
    url = "https://graph.instagram.com/me/media"
    params = {
        "fields": "id,caption,media_url,timestamp",
        "access_token": INSTAGRAM_ACCESS_TOKEN,
        "limit": 10  # Limit to most recent posts
    }

    response = requests.get(url, params=params)
    media_data = response.json().get('data', [])
    
    comment_count = 0
    
    # Process each media post
    for media in media_data:
        media_id = media.get('id')
        if not media_id:
            continue
            
        # Get comments for this media post
        comments_url = f"https://graph.instagram.com/{media_id}/comments"
        comments_params = {
            "fields": "text,timestamp",
            "access_token": INSTAGRAM_ACCESS_TOKEN,
            "limit": 50  # Fetch up to 50 comments per post
        }
        
        comments_response = requests.get(comments_url, params=comments_params)
        comments_data = comments_response.json().get('data', [])
        
        for comment in comments_data:
            comment_count += 1
            # Yield the comment text
            yield comment_count, comment.get("text", "")


def input_builder(worker_index, worker_count, resume_state):
    return get_instagram_comments()


def clean_comment(comment):
    """
    Removes spaces and special characters from a comment
    :param comment:
    :return: clean comment
    """
    comment = comment.lower()
    comment = re.sub(pattern, lambda g: contractions[g.group(0)], comment)
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", comment).split())


def get_comment_sentiment(comment):
    """
    Determines the sentiment of a comment whether positive, negative or neutral
    :param comment:
    :return: sentiment and the comment
    """
    encoded_input = tokenizer(comment, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranked = np.argsort(scores)
    ranked = ranked[::-1]
    sentiment_class = config.id2label[ranked[0]]
    sentiment_score = scores[ranked[0]]

    return sentiment_class, comment


def output_builder1(worker_index, worker_count):
    con = st.empty()
    def write_comments(sentiment__comment):
        sentiment, comment = sentiment__comment
        con.write(f'sentiment: {sentiment}, comment: {comment}')

    return write_comments


def split_text(sentiment__text):
    sentiment, text = sentiment__text
    tokens = re.findall(r'[^\s!,.?":;0-9]+', text)
    data = [(sentiment, word) for word in tokens if word not in sw_spacy]
    return data

# Add a fold window to capture the count of words
# grouped by positive, negative and neutral sentiment
cc = SystemClockConfig()
wc = TumblingWindowConfig(length=timedelta(minutes=WINDOW_SIZE))

def count_words():
    return defaultdict(lambda:0)


def count(results, word):
    results[word] += 1
    return results


def sort_dict(key__data):
    key, data = key__data
    return ("all", {key: sorted(data.items(), key=lambda k_v: k_v[1], reverse=True)})


def join(all_words, words):
    all_words = dict(all_words, **words)
    return all_words


def join_complete(all_words):
    return len(all_words) == 3


def output_builder2(worker_index, worker_count):
    placeholder = st.empty()
    def write_to_dashboard(key__data):
        key, data = key__data
        with placeholder.container():
            fig, axes = plt.subplots(1, 3)
            i = 0
            for sentiment, words in data.items():
                # Create and generate a word cloud image:
                wc = WordCloud().generate(" ".join([" ".join([x[0],]*x[1]) for x in words]))

                # Display the generated image:
                axes[i].imshow(wc)
                axes[i].set_title(sentiment)
                axes[i].axis("off")
                axes[i].set_facecolor('none')
                i += 1
            st.pyplot(fig)

    return write_to_dashboard

if __name__ == "__main__":

    st.title("Instagram Comments Analysis")

    flow = Dataflow()
    flow.input("input", ManualInputConfig(input_builder))
    flow.map(clean_comment)
    flow.inspect(print)
    flow.map(get_comment_sentiment)
    flow.inspect(print)
    flow.capture(ManualOutputConfig(output_builder1))
    flow.flat_map(split_text)
    flow.fold_window(
        "count_words", 
        cc, 
        wc, 
        builder = count_words, 
        folder = count)
    flow.map(sort_dict)
    flow.reduce("join", join, join_complete)
    flow.inspect(print)
    flow.capture(ManualOutputConfig(output_builder2))

    search_terms = [st.text_input('Enter Instagram hashtag or keyword to analyze')]
    
    if st.button("Click to Start Analyzing Instagram Comments"):
        # Set the Instagram token from .env or let the user input it if not found
        if not INSTAGRAM_ACCESS_TOKEN:
            st.warning("No Instagram access token found in environment. Please add it to your .env file.")
            INSTAGRAM_ACCESS_TOKEN = st.text_input("Enter your Instagram access token:")
        run_main(flow)