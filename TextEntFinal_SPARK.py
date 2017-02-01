import os
import nltk
from pyspark import SparkContext
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.util import mark_negation
from nltk.tag import pos_tag
os.environ["SPARK_HOME"] = "C:/Users/Channita/Downloads/spark-1.6.1-bin-hadoop2.6"

def extract(x):
    id = x[0]
    if str(x[1]) == 'YES':
        entailment = 1
    else:
        entailment = 0
    task = 0
    if str(x[2]) == 'IE':
        task = 1
    elif str(x[2]) == 'IR':
        task = 2
    elif str(x[2]) == 'QA':
        task = 3
    elif str(x[2]) == 'SUM':
        task = 4

    text = str(x[3])
    hypothesis = str(x[4])
    text_only = text
    hypothesis_only = hypothesis
    tokens_hy = nltk.word_tokenize(hypothesis_only.lower())
    tokens_txt = nltk.word_tokenize(text_only.lower())
    lem_hy = []

    for i in tokens_hy:
        lemmatizer = WordNetLemmatizer()
        tokens = lemmatizer.lemmatize(i, pos='v')
        lem_hy.append(tokens.lower())

    if '.' in lem_hy:
        lem_hy.remove('.')
    lem_txt = []

    for i in tokens_txt:
        lemmatizer = WordNetLemmatizer()
        tokens = lemmatizer.lemmatize(i, pos='v')
        lem_txt.append(tokens.lower())
    if '.' in lem_txt:
        lem_txt.remove('.')

    stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
                 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should','now', 'd', 'll', 'm',
                 'o', 're', 've', 'y', 'ain', 'ma']

    for word in stopWords:
        if word.lower() in lem_hy:
            lem_hy.remove(word)
    for word in stopWords:
        if word.lower() in lem_txt:
            lem_txt.remove(word)

    found_counter = 0
    found_counter_N = 0
    bigram_counter = 0
    trigrams_counter = 0
    noun_counter = 0
    pnoun_counter = 0
    verb_counter = 0
    adjective_counter = 0
    adverb_counter = 0
    numeral_counter = 0

    for i in lem_hy:
        if i != "" and i != "." and i != "," and i != ";":
            if i in lem_txt:
                found_counter = found_counter + 1
            else:
                found_counter = found_counter

            neg_Analysis_hy = mark_negation(lem_hy)
            neg_Analysis_txt = mark_negation(lem_txt)
            for i in neg_Analysis_hy:
                if i in neg_Analysis_txt:
                    found_counter_N = found_counter_N + 1
                else:
                    found_counter_N = found_counter_N

            bigrams_H = list(nltk.bigrams(lem_hy))
            trigrams_H = list(nltk.trigrams(lem_hy))
            bigrams_T = list(nltk.bigrams(lem_txt))
            trigrams_T = list(nltk.trigrams(lem_txt))
            len_bigrams_H = len(bigrams_H)
            len_trigrams_H = len(trigrams_H)
            len_bigrams_T = len(bigrams_T)
            len_trigrams_T = len(trigrams_T)

            for k in bigrams_H:
                if k in bigrams_T:
                    bigram_counter = bigram_counter + 1
                else:
                    bigram_counter = bigram_counter
            for l in trigrams_H:
                if l in trigrams_T:
                    trigrams_counter = trigrams_counter + 1
                else:
                    trigrams_counter = trigrams_counter

            tagged_hy = nltk.pos_tag(lem_hy)
            tagged_txt = nltk.pos_tag(lem_txt)

            nouns_hy = []
            nouns_txt = []

            for i in tagged_hy:
                if i[1] == 'N' or i[1] == 'NN' or i[1] == 'NNS':
                    nouns_hy.append(i[0])
            for i in tagged_txt:
                if i[1] == 'N' or i[1] == 'NN' or i[1] == 'NNS':
                    nouns_txt.append(i[0])

            pnouns_hy = []
            pnouns_txt = []

            for i in tagged_hy:
                if i[1] == 'NNP' or i[1] == 'NNPS':
                    pnouns_hy.append(i[0])
            for i in tagged_txt:
                if i[1] == 'NNP' or i[1] == 'NNPS':
                    pnouns_txt.append(i[0])

            numeral_hy = []
            numeral_txt = []

            for i in tagged_hy:
                if i[1] == 'CD':
                    numeral_hy.append(i[0])
            for i in tagged_txt:
                if i[1] == 'CD':
                    numeral_txt.append(i[0])

            adjective_hy = []
            adjective_txt = []

            for i in tagged_hy:
                if i[1] == 'JJ' or i[1] == 'JJR' or i[1] == 'JJS':
                    adjective_hy.append(i[0])
            for i in tagged_txt:
                if i[1] == 'JJ' or i[1] == 'JJR' or i[1] == 'JJS':
                    adjective_txt.append(i[0])

            adverb_hy = []
            adverb_txt = []

            for i in tagged_hy:
                if i[1] == 'RB':
                    adverb_hy.append(i[0])
            for i in tagged_txt:
                if i[1] == 'RB':
                    adverb_txt.append(i[0])

            verb_hy = []
            verb_txt = []

            for i in tagged_hy:
                if i[1] == 'VB' or i[1] == 'VBD' or i[1] == 'VBG' or i[1] == 'VBN' or i[1] == 'VBP' or i[1] == 'VBZ':
                    verb_hy.append(i[0])
            for i in tagged_txt:
                if i[1] == 'VB' or i[1] == 'VBD' or i[1] == 'VBG' or i[1] == 'VBN' or i[1] == 'VBP' or i[1] == 'VBZ':
                    verb_txt.append(i[0])

            for nn in nouns_hy:
                if nn in nouns_txt:
                    noun_counter = noun_counter + 1
                else:
                    noun_counter = noun_counter
            for nn in pnouns_hy:
                if nn in pnouns_txt:
                    pnoun_counter = pnoun_counter + 1
                else:
                    pnoun_counter = pnoun_counter
            for nn in numeral_hy:
                if nn in numeral_txt:
                    numeral_counter = numeral_counter + 1
                else:
                    numeral_counter = numeral_counter
            for nn in adverb_hy:
                if nn in adverb_txt:
                    adverb_counter = adverb_counter + 1
                else:
                    adverb_counter = adverb_counter
            for nn in adjective_hy:
                if nn in adjective_txt:
                    adjective_counter = adjective_counter + 1
                else:
                    adjective_counter = adjective_counter
            for nn in verb_hy:
                if nn in verb_txt:
                    verb_counter = verb_counter + 1
                else:
                    verb_counter = verb_counter

    word_neg_ratio = found_counter_N
    bigram_ratio = bigram_counter
    trigram_ratio = trigrams_counter
    word_o_ratio = found_counter
    common_nouns_ratio = noun_counter / (len(lem_txt) + len(lem_hy))
    common_pnouns_ratio = pnoun_counter / (len(lem_txt) + len(lem_hy))
    common_adverb_ratio = adverb_counter / (len(lem_txt) + len(lem_hy))
    common_adjective_ratio = adjective_counter / (len(lem_txt) + len(lem_hy))
    common_numerics_ratio = numeral_counter / (len(lem_txt) + len(lem_hy))
    common_verb_ratio = verb_counter / (len(lem_txt) + len(lem_hy))
    return [entailment, word_o_ratio, word_neg_ratio, bigram_ratio, trigram_ratio, common_nouns_ratio,
            common_pnouns_ratio, common_verb_ratio, common_adverb_ratio, common_adjective_ratio, common_numerics_ratio]

def extractTest(x):
    id = x[0]
    task = 0
    if str(x[1]) == 'IE':
        task = 1
    elif str(x[1]) == 'IR':
        task = 2
    elif str(x[1]) == 'QA':
        task = 3
    elif str(x[1]) == 'SUM':
        task = 4

    text = str(x[2])
    hypothesis = str(x[3])
    text_only = text
    hypothesis_only = hypothesis
    tokens_hy = nltk.word_tokenize(hypothesis_only.lower())
    tokens_txt = nltk.word_tokenize(text_only.lower())
    lem_hy = []

    for i in tokens_hy:
        lemmatizer = WordNetLemmatizer()
        tokens = lemmatizer.lemmatize(i, pos='v')
        lem_hy.append(tokens.lower())
    if '.' in lem_hy:
        lem_hy.remove('.')
    lem_txt = []

    for i in tokens_txt:
        lemmatizer = WordNetLemmatizer()
        tokens = lemmatizer.lemmatize(i, pos='v')
        lem_txt.append(tokens.lower())
    if '.' in lem_txt:
        lem_txt.remove('.')
    stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
                 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm',
                 'o', 're', 've', 'y', 'ain', 'ma']

    for word in stopWords:
        if word.lower() in lem_hy:
            lem_hy.remove(word)
    for word in stopWords:
        if word.lower() in lem_txt:
            lem_txt.remove(word)

    found_counter = 0
    found_counter_N = 0
    bigram_counter = 0
    trigrams_counter = 0
    noun_counter = 0
    pnoun_counter = 0
    verb_counter = 0
    adjective_counter = 0
    adverb_counter = 0
    numeral_counter = 0

    for i in lem_hy:
        if i != "" and i != "." and i != "," and i != ";":
            if i in lem_txt:
                found_counter = found_counter + 1
            else:
                found_counter = found_counter
            neg_Analysis_hy = mark_negation(lem_hy)
            neg_Analysis_txt = mark_negation(lem_txt)
            for i in neg_Analysis_hy:
                if i in neg_Analysis_txt:
                    found_counter_N = found_counter_N + 1
                else:
                    found_counter_N = found_counter_N
            bigrams_H = list(nltk.bigrams(lem_hy))
            trigrams_H = list(nltk.trigrams(lem_hy))
            bigrams_T = list(nltk.bigrams(lem_txt))
            trigrams_T = list(nltk.trigrams(lem_txt))
            len_bigrams_H = len(bigrams_H)
            len_trigrams_H = len(trigrams_H)
            len_bigrams_T = len(bigrams_T)
            len_trigrams_T = len(trigrams_T)
            for k in bigrams_H:
                if k in bigrams_T:
                    bigram_counter = bigram_counter + 1
                else:
                    bigram_counter = bigram_counter
            for l in trigrams_H:
                if l in trigrams_T:
                    trigrams_counter = trigrams_counter + 1
                else:
                    trigrams_counter = trigrams_counter
            tagged_hy = nltk.pos_tag(lem_hy)
            tagged_txt = nltk.pos_tag(lem_txt)

            nouns_hy = []
            nouns_txt = []
            for i in tagged_hy:
                if i[1] == 'N' or i[1] == 'NN' or i[1] == 'NNS':
                    nouns_hy.append(i[0])
            for i in tagged_txt:
                if i[1] == 'N' or i[1] == 'NN' or i[1] == 'NNS':
                    nouns_txt.append(i[0])

            pnouns_hy = []
            pnouns_txt = []
            for i in tagged_hy:
                if i[1] == 'NNP' or i[1] == 'NNPS':
                    pnouns_hy.append(i[0])
            for i in tagged_txt:
                if i[1] == 'NNP' or i[1] == 'NNPS':
                    pnouns_txt.append(i[0])

            numeral_hy = []
            numeral_txt = []
            for i in tagged_hy:
                if i[1] == 'CD':
                    numeral_hy.append(i[0])
            for i in tagged_txt:
                if i[1] == 'CD':
                    numeral_txt.append(i[0])

            adjective_hy = []
            adjective_txt = []
            for i in tagged_hy:
                if i[1] == 'JJ' or i[1] == 'JJR' or i[1] == 'JJS':
                    adjective_hy.append(i[0])
            for i in tagged_txt:
                if i[1] == 'JJ' or i[1] == 'JJR' or i[1] == 'JJS':
                    adjective_txt.append(i[0])

            adverb_hy = []
            adverb_txt = []
            for i in tagged_hy:
                if i[1] == 'RB':
                    adverb_hy.append(i[0])
            for i in tagged_txt:
                if i[1] == 'RB':
                    adverb_txt.append(i[0])

            verb_hy = []
            verb_txt = []
            for i in tagged_hy:
                if i[1] == 'VB' or i[1] == 'VBD' or i[1] == 'VBG' or i[1] == 'VBN' or i[1] == 'VBP' or i[1] == 'VBZ':
                    verb_hy.append(i[0])
            for i in tagged_txt:
                if i[1] == 'VB' or i[1] == 'VBD' or i[1] == 'VBG' or i[1] == 'VBN' or i[1] == 'VBP' or i[1] == 'VBZ':
                    verb_txt.append(i[0])

            for nn in nouns_hy:
                if nn in nouns_txt:
                    noun_counter = noun_counter + 1
                else:
                    noun_counter = noun_counter
            for nn in pnouns_hy:
                if nn in pnouns_txt:
                    pnoun_counter = pnoun_counter + 1
                else:
                    pnoun_counter = pnoun_counter
            for nn in numeral_hy:
                if nn in numeral_txt:
                    numeral_counter = numeral_counter + 1
                else:
                    numeral_counter = numeral_counter
            for nn in adverb_hy:
                if nn in adverb_txt:
                    adverb_counter = adverb_counter + 1
                else:
                    adverb_counter = adverb_counter
            for nn in adjective_hy:
                if nn in adjective_txt:
                    adjective_counter = adjective_counter + 1
                else:
                    adjective_counter = adjective_counter
            for nn in verb_hy:
                if nn in verb_txt:
                    verb_counter = verb_counter + 1
                else:
                    verb_counter = verb_counter

    if len(lem_hy)!= 0:
        hy_txt_length_ratio=len(lem_txt)/len(lem_hy)
    else:
        hy_txt_length_ratio=0
    word_neg_ratio = found_counter_N
    bigram_ratio = bigram_counter
    trigram_ratio = trigrams_counter
    word_o_ratio = found_counter
    common_nouns_ratio = noun_counter / (len(lem_txt) + len(lem_hy))
    common_pnouns_ratio = pnoun_counter / (len(lem_txt) + len(lem_hy))
    common_adverb_ratio = adverb_counter / (len(lem_txt) + len(lem_hy))
    common_adjective_ratio = adjective_counter / (len(lem_txt) + len(lem_hy))
    common_numerics_ratio = numeral_counter / (len(lem_txt) + len(lem_hy))
    common_verb_ratio = verb_counter / (len(lem_txt) + len(lem_hy))
    return [word_o_ratio, word_neg_ratio, bigram_ratio, trigram_ratio, common_nouns_ratio,
            common_pnouns_ratio, common_verb_ratio, common_adverb_ratio, common_adjective_ratio, common_numerics_ratio]


    #common_nouns_ratio, common_pnouns_ratio, common_verb_ratio, common_adverb_ratio, common_adjective_ratio,common_numerics_ratio

#Initializing Spark
sc = SparkContext()

#Importing File
textFile = sc.textFile("textual_entailment_development.xml")

#Getting the RDDs the right way to work
firstMap = textFile.map(lambda x: [x.split("id=\"",1)[1].split("\"",1)[0],x])
secondMap = firstMap.map(lambda x: [x[0], x[1].split("entailment=\"",1)[1].split("\"",1)[0], x[1]])
thirdMap = secondMap.map(lambda x: [x[0], x[1], x[2].split("task=\"",1)[1].split("\"",1)[0], x[2]])
fourthMap = thirdMap.map(lambda x: [x[0], x[1], x[2], x[3].split("<t>",1)[1].split("</t>",1)[0], x[3]])
fifthMap = fourthMap.map(lambda x: [x[0], x[1], x[2], x[3], x[4].split("<h>",1)[1].split("</h>",1)[0]])
#Executing function on the RDD
finalMap = fifthMap.map(lambda x: extract(x))

# Mapping
from pyspark.mllib.regression import LabeledPoint
trainingData = finalMap.map(lambda x: LabeledPoint(x[0],x[1:]))

# for each in trainingData.take(5):
#     print(each)

# #SUPPORT VECTOR MACHINE
# from pyspark.mllib.classification import SVMWithSGD, SVMModel
#
# # Build the model
# model = SVMWithSGD.train(trainingData, iterations=100)
#
# # Evaluating the model on training data
# labelsAndPreds = trainingData.map(lambda p: (p.label, model.predict(p.features)))
# trainErr = labelsAndPreds.filter(lambda x: x[0] != x[1]).count() / float(trainingData.count())
# print("SVM Training Error = " + str(trainErr))

#RANDOM FOREST
from pyspark.mllib.tree import RandomForest
print("Training the model on the 70% of training data")

(trainingSet, testSet) = trainingData.randomSplit([0.7, 0.3])

model1 = RandomForest.trainClassifier(trainingSet, numClasses=2, categoricalFeaturesInfo={},numTrees=3,
                                      featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)

trainpredictions = model1.predict(trainingSet.map(lambda x: x.features))
trainlabelsAndPredictions = trainingSet.map(lambda lp: lp.label).zip(trainpredictions)
trainErr = trainlabelsAndPredictions.filter(lambda x: x[0] != x[1]).count() / float(trainingSet.count())
print('Training Error = ' + str(trainErr))

testpredictions = model1.predict(testSet.map(lambda x: x.features))
labelsAndPredictions = testSet.map(lambda lp: lp.label).zip(testpredictions)
testErr = labelsAndPredictions.filter(lambda x: x[0] != x[1]).count() / float(testSet.count())
print('Test Error = ' + str(testErr))

print("\nTraining the model on the full training data")
trainingSet = trainingData
model1 = RandomForest.trainClassifier(trainingSet, numClasses=2, categoricalFeaturesInfo={}, numTrees=3,
                                      featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)
trainpredictions = model1.predict(trainingSet.map(lambda x: x.features))
trainlabelsAndPredictions = trainingSet.map(lambda lp: lp.label).zip(trainpredictions)
trainErr = trainlabelsAndPredictions.filter(lambda x: x[0] != x[1]).count() / float(trainingSet.count())
print('Full Training Error = ' + str(trainErr))

# PREDICTING ON THE TEST SET
textFileTest = sc.textFile("textual entailment test.xml")
firstMapTest = textFileTest.map(lambda x: [x.split("id=\"",1)[1].split("\"",1)[0],x])
secondMapTest = firstMapTest.map(lambda x: [x[0], x[1].split("task=\"",1)[1].split("\"",1)[0], x[1]])
thirdMapTest = secondMapTest.map(lambda x: [x[0], x[1], x[2].split("<t>",1)[1].split("</t>",1)[0], x[2]])
fourthMapTest = thirdMapTest.map(lambda x: [x[0], x[1], x[2], x[3].split("<h>",1)[1].split("</h>",1)[0]])
finalMapTest = fourthMapTest.map(lambda x: extractTest(x))
testingData = finalMapTest.map(lambda x: LabeledPoint(0, x))

for each in finalMapTest.take(5):
    print(each)
predictions = model1.predict(testingData.map(lambda x: x.features)).zip(textFileTest)

def n_to_str(x):
    if x == 0.0:
        return 'NO'
    else:
        return 'YES'

fifthMapTest = predictions.map(lambda x: [n_to_str(x[0]), x[1]])
for each in fifthMapTest.take(5):
    print(each)

sixthMapTest = fifthMapTest.map(lambda x: [x[0], x[1].split("task=\"",1)[0], x[1].split("task=\"",1)[1]])
seventhMapTest = sixthMapTest.map(lambda x: [x[0], x[1]+str("entailment=\""), x[2]])
eightMapTest = seventhMapTest.map(lambda x: [x[1]+str(x[0]), x[2]])

ninthMapTest = eightMapTest.map(lambda x: [x[0]+str("\" task=\"")+x[1]])
for each in ninthMapTest.take(5):
    print(each)
print("ninthMapTest")

tenthMapTest = ninthMapTest.map(lambda record: '\t'.join(str(record)))
for each in ninthMapTest.take(796):
    print(each)
df = ninthMapTest.saveAsTextFile("test")


