import pandas
import tweepy
import csv
from tweepy import OAuthHandler

access_token = "Twitter API KEY"
access_token_secret = "Twitter API KEY"
consumer_key = "Twitter API KEY"
consumer_secret = "Twitter API KEY"

authAPI = OAuthHandler(consumer_key, consumer_secret)
authAPI.set_access_token(access_token, access_token_secret)
API = tweepy.API(authAPI)

streamErrorCount = 1

class StreamListener(tweepy.StreamListener):

    def on_status(self, status):
        global streamErrorCount
        fo = open("/Users/UP719920/Documents/Final Year Project/projectFiles/accountStream.csv", "a")
        writer = csv.writer(fo)
        text = status.text
        name = status.user.screen_name
        if status.retweeted == False:
            try:
                writer.writerow([text.encode("utf-8"), name.encode("utf-8")])
            except Exception as e:
                print("twitter stream error: ", streamErrorCount, e)
                streamErrorCount = streamErrorCount + 1

    def on_error(self, statusCode):
        if statusCode == 420:
            return False
        print(statusCode)

twitterStreamListener = StreamListener()
twitterStream = tweepy.Stream(auth=authAPI, listener=twitterStreamListener)
twitterStream.filter(follow=["Twitter Account Number"])
