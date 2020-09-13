import requests
import json
import re
from requests_oauthlib import OAuth1

CONSUMER_KEY = 'RKDPXOaXJUJsJlgKbDRbkp2T3'
CONSUMER_SECRET = '6lv04ltMXeS5jMdixQDbB5naSl4k7Pm7d6qmXv9agUHWVKxtD3'
ACCESS_TOKEN = '901893222141632512-Lac1pHUaZwD3RVYz1lA8i2KVMEvWn70'
ACCESS_SECRET = 'MRQIYCfO9pndI2ktiNIz8arztJ8IWZMzpluprYEnmpJum'

API_SEARCH_URL = 'https://api.twitter.com/1.1/search/tweets.json'
QUERY = '?q=%23{}&src=tyah&count=100'

QUERY_TERMS = 'FantasyFootball'

CSV_PATH = '/Users/changlonghuang/Documents/Python/Twitter/2017_fantasy_football_tweets.csv'

def get_auth():
    """
    Returns the authorization key to input into the request field | auth=
    """
    auth = OAuth1(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
    return auth

def write_twitter_info_csv(twitter_list, csv_1):
    with open(csv_1, 'a') as csvfile:
        tinder_write = csv.writer(csvfile)
        try:
            tinder_write.writerow(twitter_list)
        except:
            print "Error writing row for", ' '.join(twitter_list)
    return "This csv write function worked"

def main():
    
    r = requests.get(url=API_SEARCH_URL + QUERY.format(QUERY_TERMS), auth=get_auth())
    print r.status_code
    response_json = r.json()
    search_metadata = response_json.values()[0]
    list_of_tweets = response_json.values()[1]

    try:
        next_results_query = search_metadata['next_results']
    except:
        print 'No next result from query'
    max_id = search_metadata['max_id']
    since_id = search_metadata['since_id']
    
    # I should create a function to parse hashtags and remove advertising text
    # maybe regex

    for tweet in list_of_tweets:
        tweet_id = tweet['id']
        tweet_text = tweet['text']
        # broken_tweet_text = remove_advertising_text(tweet_text)
        user = tweet['user']
        twitter_user_id = user['id']
        followers_count = user['followers_count']
        friends_count = user['friends_count']
        location = user['location']
        retweet_boolean = tweet['retweeted']
        language = tweet['lang']
        geography = tweet['geo']
        tweet_list = [tweet_id, tweet_text, twitter_user_id, followers_count,
                     friends_count, location, retweet_boolean, language, geography]
        print tweet_text
        # write_twitter_info_csv(tweet_list, CSV_PATH)

def remove_advertising_text(json_output):
    # get anything before #
    # get anything after #
    try:
        tweet_text = re.search('^(.+?)\#| ^(.+?)\http', json_output)
        return (tweet_text.group(0))
    except:
        return json_output


if __name__ == "__main__":
    main()
    