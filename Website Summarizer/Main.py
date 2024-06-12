from bs4 import BeautifulSoup
import requests
from math import sqrt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist

#Website preperation
url = input("Enter the website: ")
res = requests.get(url)
soup = BeautifulSoup(res.text,'html.parser')

#Creates a string variable containing visable website text

rowContent = ''
for i in range(len(soup.select('p'))):
    rowContent += soup.select('p')[i].getText()

#Tokenizing the page content
sentences = sent_tokenize(rowContent)
words = word_tokenize(rowContent.lower())

#Removing stop words
stopWords = set(stopwords.words("english"))
filteredWords = [word for word in words if word.casefold() not in stopWords]

#Frequency distribution
frequencyDistribution = FreqDist(filteredWords)

#Assigning scores to sentences for frequencies
sentenceScores = {}
for i, sentence in enumerate(sentences):
    for word in word_tokenize(sentence.lower()):
        if word in frequencyDistribution:
            if i in sentenceScores:
                sentenceScores[i] += frequencyDistribution[word]
            else:
                sentenceScores[i] = frequencyDistribution[word]

#Sort sentences by scores in descending order   
sortedSentences = sorted(sentenceScores, key=lambda x: sentenceScores[x], reverse=True)

#Using the number of sentences varauble to formulate the final summary
summarySentences = sorted(sortedSentences[:round(len(sentences)*.25)])

#Actually creates the summary
summary = ' '.join([sentences[i] for i in summarySentences])

print(summary)