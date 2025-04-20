import re
from datetime import timedelta
from collections import defaultdict

from bytewax.dataflow import Dataflow
from bytewax.inputs import ManualInputConfig
from bytewax.outputs import StdOutputConfig, ManualOutputConfig
from bytewax.execution import run_main
from bytewax.window import TumblingWindowConfig, SystemClockConfig

import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import json
import requests
from dotenv import load_dotenv
import os
import openai
from datetime import datetime
from contractions import contractions

# load environment variables
load_dotenv()

# load spacy stop words
en = spacy.load('en_core_web_sm')
en.Defaults.stop_words |= {"s","t",}
sw_spacy = en.Defaults.stop_words

# load contractions and compile regex
pattern = re.compile(r'\b(?:{0})\b'.format('|'.join(contractions.keys())))

# Set up OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    using OpenAI API
    :param comment:
    :return: sentiment and the comment
    """
    if not comment.strip():
        return "neutral", comment
        
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis tool. Analyze the sentiment of the following text and respond with exactly one word: 'positive', 'negative', or 'neutral'."},
                {"role": "user", "content": comment}
            ],
            max_tokens=10,
            temperature=0
        )
        
        # Extract the sentiment from the response
        sentiment = response.choices[0].message.content.strip().lower()
        
        # Ensure we only return valid sentiment classes
        if sentiment not in ["positive", "negative", "neutral"]:
            sentiment = "neutral"
            
        return sentiment, comment
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return "neutral", comment


def output_builder1(worker_index, worker_count):
    # Initialize session state for comment history if it doesn't exist
    if 'comment_history' not in st.session_state:
        st.session_state.comment_history = []

    # Create a single container for the output
    output_container = st.empty()

    def write_comments(sentiment__comment):
        sentiment, comment = sentiment__comment

        # Add new comment to history (keep only last 50 comments)
        st.session_state.comment_history.insert(0, {  # Add newest at beginning
            "sentiment": sentiment,
            "comment": comment,
            "time": datetime.now().strftime("%H:%M:%S")
        })
        if len(st.session_state.comment_history) > 50:
            st.session_state.comment_history = st.session_state.comment_history[:50]

        result = ""
        for item in st.session_state.comment_history:
            result += f"""
                 {item['sentiment']}  | {item['comment']}
            """

        result += f""" Total comments: {len(st.session_state.comment_history)}"""

        # Update the container
        with output_container.container():
            st.markdown(result)

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


if __name__ == "__main__":

    st.title("Instagram Comments Analysis")
    
    # Initialize comment history in session state if not exists
    if 'comment_history' not in st.session_state:
        st.session_state.comment_history = []
        
    # Display history table if there are comments
    if st.session_state.comment_history:
        with st.expander("View Comment History", expanded=True):
            history_df = pd.DataFrame(st.session_state.comment_history)
            st.dataframe(history_df, use_container_width=True)

    flow = Dataflow()
    flow.input("input", ManualInputConfig(input_builder))
    flow.map(clean_comment)
    flow.inspect(print)
    flow.map(get_comment_sentiment)
    flow.inspect(print)
    flow.capture(ManualOutputConfig(output_builder1))


    search_terms = [st.text_input('Enter Instagram hashtag or keyword to analyze')]
    
    if st.button("Click to Start Analyzing Instagram Comments"):
        # Check for required API keys
        if not openai.api_key:
            st.warning("No OpenAI API key found in environment. Please add it to your .env file as OPENAI_API_KEY.")
            openai.api_key = st.text_input("Enter your OpenAI API key:", type="password")
            if not openai.api_key:
                st.error("OpenAI API key is required for sentiment analysis.")
                st.stop()
                
        # Set the Instagram token from .env or let the user input it if not found
        if not INSTAGRAM_ACCESS_TOKEN:
            st.warning("No Instagram access token found in environment. Please add it to your .env file.")
            INSTAGRAM_ACCESS_TOKEN = st.text_input("Enter your Instagram access token:")
            if not INSTAGRAM_ACCESS_TOKEN:
                st.error("Instagram access token is required to fetch comments.")
                st.stop()
                
        run_main(flow)