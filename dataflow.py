import re
from datetime import timedelta, datetime
from collections import defaultdict
import signal
import sys

from bytewax.dataflow import Dataflow
from bytewax.inputs import ManualInputConfig
from bytewax.outputs import StdOutputConfig, ManualOutputConfig
from bytewax.execution import run_main
from bytewax.window import TumblingWindowConfig, SystemClockConfig

import spacy
import streamlit as st
import requests
from dotenv import load_dotenv
import os
import openai
from contractions import contractions

# Global flag for stopping
should_stop = False

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
    """Get comments from Instagram"""
    global should_stop
    INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")

    # Get recent media posts first
    url = "https://graph.instagram.com/me/media"
    params = {
        "fields": "id,caption,media_url,timestamp",
        "access_token": INSTAGRAM_ACCESS_TOKEN,
        "limit": 10
    }

    response = requests.get(url, params=params)
    media_data = response.json().get('data', [])

    comment_count = 0

    # Process each media post
    for media in media_data:
        if should_stop:
            break

        media_id = media.get('id')
        if not media_id:
            continue

        # Get comments for this media post
        comments_url = f"https://graph.instagram.com/{media_id}/comments"
        comments_params = {
            "fields": "text,timestamp",
            "access_token": INSTAGRAM_ACCESS_TOKEN,
            "limit": 50
        }

        comments_response = requests.get(comments_url, params=comments_params)
        comments_data = comments_response.json().get('data', [])

        for comment in comments_data:
            if should_stop:
                break

            comment_count += 1
            yield comment_count, comment.get("text", "")

def input_builder(worker_index, worker_count, resume_state):
    for comment_count, comment_text in get_instagram_comments():
        yield comment_count, comment_text
        if should_stop:
            break

def clean_comment(comment):
    """Clean the comment text"""
    comment = comment.lower()
    comment = re.sub(pattern, lambda g: contractions[g.group(0)], comment)
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", comment).split())

def get_comment_sentiment(comment):
    """Get sentiment using OpenAI API"""
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

        sentiment = response.choices[0].message.content.strip().lower()
        return sentiment if sentiment in ["positive", "negative", "neutral"] else "neutral", comment
    except Exception:
        return "neutral", comment

def output_builder1(worker_index, worker_count):
    if 'comment_history' not in st.session_state:
        st.session_state.comment_history = []

    # Create containers for both live view and history
    live_container = st.empty()
    history_container = st.container()

    def write_comments(sentiment__comment):
        sentiment, comment = sentiment__comment

        # Add new comment to history
        st.session_state.comment_history.insert(0, {
            "sentiment": sentiment,
            "comment": comment,
            "time": datetime.now().strftime("%H:%M:%S")
        })

        # Update live view
        with live_container.container():
            st.subheader("Live Comments")
            if st.session_state.comment_history:
                latest = st.session_state.comment_history[0]
                st.write(f"**Latest: {latest['sentiment'].upper()}** ({latest['time']}): {latest['comment']}")
                st.write(f"Total comments collected: {len(st.session_state.comment_history)}")
            else:
                st.write("No comments collected yet")

        # Always show full history (will be visible after stopping)
        with history_container:
            st.subheader("Comment History")
            if st.session_state.comment_history:
                for item in st.session_state.comment_history:
                    st.write(f"**{item['sentiment'].upper()}** ({item['time']}): {item['comment']}")
            else:
                st.write("No comments in history")

    return write_comments


if __name__ == "__main__":
    st.title("Instagram Comments Analysis")

    # Initialize session state
    if 'comment_history' not in st.session_state:
        st.session_state.comment_history = []

    flow = Dataflow()
    flow.input("input", ManualInputConfig(input_builder))
    flow.map(clean_comment)
    flow.map(get_comment_sentiment)
    flow.capture(ManualOutputConfig(output_builder1))

    col1, col2 = st.columns(2)

    if col1.button("Start Analysis"):
        # [Your existing start logic...]
        should_stop = False
        run_main(flow)

    if col2.button("Stop Analysis"):
        should_stop = True
        st.success(f"Stopped. Collected {len(st.session_state.comment_history)} comments.")

        # Display full history after stopping
        st.subheader("All Collected Comments")
        if st.session_state.comment_history:
            for item in st.session_state.comment_history:
                st.write(f"**{item['sentiment'].upper()}** ({item['time']}): {item['comment']}")
        else:
            st.write("No comments were collected")