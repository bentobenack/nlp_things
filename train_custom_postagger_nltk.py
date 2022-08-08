# Train a custom pos tagger with nltk using dictionary
# NLTK pos_tag() - http://www.nltk.org/book/ch05.html

# Imports
import nltk
import pickle

# Train Data
def sampleData():
    return [
        "New York is the capital of United States.",
        "Steve Jobs was the CEO of Apple.",
        "iPhone was Invented by Apple.",
        "Books can be purchased in Market.",
    ]

# Create a Dictionary
def buildDictionary():
    dictionary = {}
    for sent in sampleData():
        partsOfSpeechTags = nltk.pos_tag(nltk.word_tokenize(sent))
        for tag in partsOfSpeechTags:
            value = tag[0]
            pos = tag[1]
            dictionary[value] = pos
    print("\nCreated Dictionary:")
    print(dictionary)        
    return dictionary

# Fucntion to save the POS Tagger
def saveMyTagger(tagger, fileName):
    fileHandle = open(fileName, "wb")
    pickle.dump(tagger, fileHandle)
    fileHandle.close()

# Function to train a custom tagger using a diccionary
def Training(fileName):
    tagger = nltk.UnigramTagger(model=buildDictionary())
    saveMyTagger(tagger, fileName)

# Load the POS Tagger
def loadMyTagger(fileName):
    return pickle.load(open(fileName, "rb"))

# Sentence
sentence = 'iPhone is purchased by Steve Jobs in New York Market'

# Tagger name
fileName = "myTagger.pickle"

# Call the function to train the tagger
Training(fileName)

# Load the trained tagger
myTagger = loadMyTagger(fileName)

# Using the trained POS Tagger
print("\nUsing the trained POS Tagger")
print(myTagger.tag(nltk.word_tokenize(sentence)))
print("\n")
