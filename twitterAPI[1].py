from twitter import *
from io import open
import csv
import json

#ACCESS_KEY,ACCESS_SECRET,CONSUMER_KEY,CONSUMER_SECRET
twitter = Twitter(auth = OAuth("Twitter API KEY", "Twitter API KEY", "Twitter API KEY","Twitter API KEY"))
posts=twitter.statuses.user_timeline(screen_name="Account Name", count=150) #number of tweets

#store data in lists and then save lists to csv
tweets=[]
dates=[]

fo = open("/Users/UP719920/Documents/Final Year Project/projectFiles/negThreatTweets.csv","ab") #ab-append binary, wb-write binary
writer=csv.writer(fo)

for p in posts:
    writer.writerow([p["text"].encode("utf-8")])

fo.close()

#-------------------------------------------------------------------
#to alternatively save twitter dataset to a JSON file
#with open("/Users/UP719920/Documents/Final Year Project/projectFiles/posThreatTweets.json", mode="w", encoding="utf-8") as fo:
#    json.dump(posts,fo)
