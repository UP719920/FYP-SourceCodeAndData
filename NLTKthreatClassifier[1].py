import nltk
import pandas as pnds

#imports names and stopwords from nltk corpus
from nltk.corpus import stopwords, names
#imports two calssifers from nltk classify
from nltk.classify import SklearnClassifier, NaiveBayesClassifier

#imports porter stemmer used to stem words to base strings
from nltk.stem import PorterStemmer
#imports tokeinzer to turn text in codes
from nltk.tokenize import PunktSentenceTokenizer

#reads csv file of threatening tweets
data = pnds.read_csv("/Users/UP719920/Documents/Final Year Project/projectFiles/posThreatTweetsMod.csv")
#saves only text from csv to data variable
data = data[['text', 'sentiment']]

tweets = []
stopwordsList = set(stopwords.words('english'))

for index, line in data.iterrows():
    filteredData = [word.lower() for word in line.text.split() if len(word) >= 3]
    cleanedData = [word for word in filteredData if 'http' not in word
    and not word.startswith('@')
    and not word.startswith('#')
    and word != 'RT']
    stopwordFreeWords = [word for word in cleanedData if not word in stopwordsList]
    tweets.append((stopwordFreeWords))

#creates a dictionary of training words
def featuresOfWords(inputWords):
    return dict([(inputWords, True)])

#imports list of threatening words
threateningVocabulary = open("/Users/UP719920/Documents/Final Year Project/projectFiles/threatVocab.rtf", "r")
threatFeatures = [(featuresOfWords(threat[:-2]), "threat") for threat in threateningVocabulary]

threatClassifier = NaiveBayesClassifier.train(threatFeatures)
threatScore = []
tweetCount = 0
totalThreat = 0

for tweet in tweets:
    tweetThreatScore = 0
    for word in tweet:
        threatResult = threatClassifier.classify(featuresOfWords(word))
        print(featuresOfWords(word))
        print(threatResult)
        if threatResult == "threat":
            tweetThreatScore += 1
    threatScore.append(tweetThreatScore)
    tweetCount += 1
    totalThreat += tweetThreatScore
    
#create text file of threat scores for threatening tweets
threatScoreFile = open("/Users/UP719920/Documents/Final Year Project/projectFiles/posThreatScoreFile2.txt", "w")
for score in threatScore:
    threatScoreFile.write(str(score) + '\n')
    
    
    
#----------------------------------------------------------------------------
#Additional tweet cleaning tagging/tokenizing methods

#break down tweets into list of words
def wordsFromTweets(tweets):
    tweetWords = []
    for words in tweets:
        tweetWords.extend(words)
    return tweetWords

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features

#wordFeatures = get_word_features(wordsFromTweets(tweets))

stemmedTweets = []

def stemWords(tweetWords):
    portStem = PorterStemmer()
    for word in tweetWords:
        try:
            stemmedTweets.append(portStem.stem(word))
        except:
            print("Can't stem word")
    return stemmedTweets

wordTokenCodes = []

def speechTag(tweetWords):
    for word in tweetWords:
        wordTokenCodes.append(nltk.pos_tag(nltk.word_tokenize(word)))
    return wordTokenCodes
