import json, re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib

# Save model to disk.
from gensim.test.utils import datapath
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import re, pickle
import time
from nltk import FreqDist
from scipy.stats import entropy
import pandas as pd
import numpy as np
from scipy.stats import entropy
from gensim import corpora, models, similarities

app = Flask(__name__)


#load target data
loaded_target = joblib.load('models/target.sav')
stopwords = joblib.load('models/stopwords.sav')
# Stem word tokens and remove stop words
stemmer_eng = SnowballStemmer("english", ignore_stopwords=True)
stemmer_germ = SnowballStemmer("german", ignore_stopwords=True) 


# Get Dictionary of the Trained Data
with open('models/dictionary', 'rb') as data:
    dictionary = pickle.load(data)

# Get Projects data
with open('models/APP_DATA.sav', 'rb') as data:
    projects = pickle.load(data)

#Get corpus 
with open('models/corpus', 'rb') as data:
    corpus = pickle.load(data)

# Load a potentially pretrained model from disk.
#lda_model = LdaModel.load('models/model_tm', mmap='r')

# later on, load trained model from file
lda_model =  models.LdaModel.load('models/lda.model')
all_topic_distr_list = lda_model[corpus]
def tokenize(text):
    """Normalize, tokenize and stem text string
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    stemmed: list of strings. List containing normalized and stemmed word tokens
    """

    try:
        # Convert text to lowercase and remove punctuation
        text = re.sub("[^a-zA-Z ]", " ", text.lower()) #remove non alphbetic text
        # Tokenize words
        tokens = word_tokenize(text)
        stemmed = [stemmer_germ.stem(word) for word in tokens if word not in stopwords]
        stemmed = [stemmer_eng.stem(word) for word in stemmed if len(word) > 1]
        stemmed = [word for word in stemmed if word not in stopwords]
    except IndexError:
        pass

    return stemmed
def clean_lower_tokenize(text):
    """
    Function to clean, lower and tokenize texts
    Returns a list of cleaned and tokenized text
    """
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)  #remove websites texts like email, https, www
    text = re.sub("[^a-zA-Z ]", "", text) #remove non alphbetic text
    text = text.lower() # lower case the text
    text = nltk.word_tokenize(text)
    return text

def remove_stop_words(text):
    """
    Function that removes all stopwords from text
    """
    return [word for word in text if word not in stopwords]

def stem_eng_german_words(text):
    """
    Function to stem words
    """
    try:
        text = [stemmer_germ.stem(word) for word in text]
        #text = [stemmer_eng.stem(word) for word in text]
        text = [word for word in text if len(word) > 1] 
    except IndexError:
        pass
    return text

def all_processing(text):
    """
    This function applies all the functions above into one
    """
    return stem_eng_german_words(remove_stop_words(clean_lower_tokenize(text)))
# load model
# load model
model = joblib.load("models/model.pkl")
def get_category(text, model, labels):
    predicted = model.predict([text])[0]
    results = dict(zip(labels.columns, predicted))
    return results

def percentage(data):
    total = np.sum(data)
    perc_arr = np.array([(x/total)*100 for x in data])
    return perc_arr
# index webpage displays cool visuals and receives user input text for model

def predict_bereich(text, lda_model):
    clean = all_processing(text)
    text_bow = dictionary.doc2bow(clean)
    topic_distr_array = np.array([topic[1] for topic in lda_model.get_document_topics(bow=text_bow)])
    labels_array_percent = percentage(topic_distr_array)
    labels_array =labels_array_percent.argsort()[-2:][::-1]
    print(topic_distr_array)
    return labels_array, labels_array_percent
def js_similarity_score(doc_distr_query, corpus_distr):
    """
    This function finds the similarity score of a given doc accross all docs in the corpus
    It takes two parameters: doc_distr_query and corpus_distr
    (1) doc_distr_query is the input document query which is an LDA topic distr: list of floats (series)
            [1.9573441e-04,...., 2.7876711e-01]
    (2) corpus_dist is the target corpus containing the LDA topic distr of all documents in the corpus: lists of lists of floats (vector)
            [[1.9573441e-04, 2.7876711e-01, 1.9573441e-04]....[1.9573441e-04,...., 2.7876711e-01]]
    It returns an array containing the similarity score of each document in the corpus_dist to the input doc_distr_query
    The output looks like this: [0.3445, 0.35353, 0.5445,.....]
    
    """
    input_doc = doc_distr_query[None,:].T #transpose input
    corpus_doc = corpus_distr.T # transpose corpus
    m = 0.5*(input_doc + corpus_doc)
    sim_score = np.sqrt(0.5*(entropy(input_doc,m) + entropy(corpus_doc,m)))
    return sim_score
def find_top_similar_docs(doc_distr_query, corpus_distr,n=10):
    """
    This function returns the index lists of the top n most similar documents using the js_similarity_score
    n can be changed to any amount desired, default is 10
    """
    sim_score = js_similarity_score(doc_distr_query, corpus_distr)
    similar_docs_index_array = sim_score.argsort()[:n] #argsort sorts from lower to higher
    return similar_docs_index_array

def recommend(text):
    clean = all_processing(text)
    text_bow = dictionary.doc2bow(clean)
    new_doc_distribution = np.array([tup[1] for tup in lda_model.get_document_topics(bow=text_bow)])
    corpus_topic_dist= np.array([[topic[1] for topic in docs] for docs in all_topic_distr_list])
    similar_docs_index = find_top_similar_docs(new_doc_distribution, corpus_topic_dist)
    top_sim_doc = projects[projects.index.isin(similar_docs_index)]
    PROJECT_DICT = top_sim_doc.to_dict() 
    return PROJECT_DICT


@app.route("/")
@app.route('/home')
def home():
    # save user input in query
    query = request.args.get('query', '')
    labels, labels_perc = predict_bereich(query, lda_model)
    recommended_projects = recommend(query)
    print(recommended_projects)
    # use model to predict classification for query
    output = get_category(query, model, loaded_target)
    print(labels)
    index_highest = labels_perc.argmax()
    group = ['ERP/SAP','SW_Dev/Enwickung','IT_App_Mgr/SW_Dev_Arch','SW_Dev/DevOps','Sys_Admin/Support', 'IT_Admin_SW/Oracle/Ops','Data/Ops','IT_Process_Mgr/Consultant', 'MS_DEV/Admin','Business_Analyst/Consulting']
    #dictOfWords = { i : other_labels[i] for i in range(0, len(other_labels) ) }
    # This will render the go.html Please see that file. 
    if labels_perc[index_highest] < 20:
        query = 'Please enter a valid text'
        return render_template(
        'home.html',
        query=query,
        output={},
        labels={},
        group={}
    )
    else:
        
        return render_template(
            'home.html',
            query=query,
            output=output,
            labels=labels,
            group=group,
            topic_distr=labels_perc,
            recommended_projects=recommended_projects
        )

@app.route("/about/")
def about():
    return render_template("about.html")

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()