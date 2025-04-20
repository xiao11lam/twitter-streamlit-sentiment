import os
import requests
import json

# To set your enviornment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = "AAAAAAAAAAAAAAAAAAAAAET10gEAAAAAK55E14t%2FSRlEZ4V7Pk97IeQh848%3DZZ2pdIpdQ5Xz6CRkJ5XurOlq1Bhr9dGBf5kdzjEpXBfE7vgvzw"
# GRJyZFnucAo7vxq4xgWbd0DVF
#AAAAAAAAAAAAAAAAAAAAAET10gEAAAAAXfFae0sLACemOvb%2BXsYiZysnkpE%3DzE7KjmyNVPZVE0pAFvoxvcRF5v04jDkl2sOtHYH7CrzQraW5R9
def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FilteredStreamPython"
    return r


def get_rules():
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream/rules", auth=bearer_oauth
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot get rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print(json.dumps(response.json()))
    return response.json()



def get_stream():
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream", auth=add_bearer_oauth, stream=True,
    )
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Cannot get stream (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    tweetnbr=0
    for response_line in response.iter_lines():
        if response_line:
            json_response = json.loads(response_line)
            tweet = json_response["data"]["text"]
            tweetnbr+=1
            yield tweetnbr, str(tweet)

def add_bearer_oauth(req):
    """
    Method required by bearer token authentication for Twitter API
    :param req:
    :return: a request object with additional headers
    """
    req.headers["Authorization"] = f"Bearer {bearer_token}"
    req.headers["User-Agent"] = "v2FilteredStreamPython"
    return req

def set_stream_rules(search_terms):
    """
    Set the rules of the stream that we want the API to return
    :param search_terms: List of all the hashtags or keywords we want to search
    :return: Json response of the set rules
    """
    # only get original tweets in english
    default_rules = 'followers_count:50 -is:retweet -is:reply is:verified -is:nullcast lang:en'
    search_rules = []
    for search_term in search_terms:
        search_rules.append({"value": f"{search_term} {default_rules}"})
    payload = {"add": search_rules}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        auth=add_bearer_oauth,
        json=payload,
    )
    if response.status_code != 201:
        raise Exception("Cannot add rules (HTTP {}): {}".format(response.status_code, response.text))
    print(json.dumps(response.json()))