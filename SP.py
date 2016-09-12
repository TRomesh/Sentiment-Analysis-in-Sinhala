import nltk
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

postrain = __location__ + "\\train\\positive\\positive.txt"
negtrain = __location__ + "\\train\\negitive\\negative.txt"
strain = __location__ + "\\train\\stopwords\\stopwords.txt"

neglist = []
poslist = []
stlist = []


#Load positive sentences into a list
pw = open(postrain, encoding="utf8")
postxt = pw.readlines()

#Load negative sentences into a list
nw = open(negtrain, encoding="utf8")
negtxt = nw.readlines()

#Load stop words into a list
sw = open(postrain, encoding="utf8")
stptxt = sw.readlines()


#Create a list of 'positive' with the exact length of our negative sentences list.
for i in range(0,len(postxt)):
    poslist.append('positive')

#Create a list of 'negatives' with the exact length of our negative sentences list.
for i in range(0,len(negtxt)):
    neglist.append('negative')

#Likewise for stopwords.
for i in range(0,len(stptxt)):
    stlist.append('stop')

#Creates a list of tuples, with sentence tagged.
postagged = zip(postxt, poslist)
negtagged = zip(negtxt, neglist)

#Combines all of the tagged sentence to one large list.
taggedsentence = postagged + negtagged

sentences = []

#Create a list of words in the sentence, within a tuple.
for (word, sentiment) in taggedsentence:
    word_filter = [i.lower() for i in word.split()]
    sentences.append((word_filter, sentiment))

#Pull out all of the words in a list of tagged sentence, formatted in tuples.
def getwords(sentences):
    allwords = []
    for (words, sentiment) in sentences:
        allwords.extend(words)
    return allwords

#Order a list of tweets by their frequency.
def getwordfeatures(listofsentences):
    #Print out wordfreq if you want to have a look at the individual counts of words.
    wordfreq = nltk.FreqDist(listofsentences)
    words = wordfreq.keys()
    return words

#Calls above functions - gives us list of the words in the sentence, ordered by freq.
print(getwordfeatures(getwords(sentences)))

wordlist = getwordfeatures(getwords(sentences))
wordlist = [i for i in wordlist if not i in stptxt]

def feature_extractor(doc):
    docwords = set(doc)
    features = {}
    for i in wordlist:
        features['contains(%s)' % i] = (i in docwords)
    return features

#Creates a training set - classifier learns distribution of true/falses in the input.
training_set = nltk.classify.apply_features(feature_extractor, sentences)
classifier = nltk.NaiveBayesClassifier.train(training_set)

print(classifier.show_most_informative_features(n=30))

while True:
    input = raw_input("Enter any sentence or 'exit' to quit:")
    if input == 'exit':
        break
    elif input == 'informfeatures':
        print(classifier.show_most_informative_features(n=30))
        continue
    else:
        input = input.lower()
        input = input.split()
        print('\nWe think that the sentiment was ' + classifier.classify(feature_extractor(input)) + ' in that sentence.\n')

pw.close()
nw.close()
sw.close()

