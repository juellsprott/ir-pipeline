import regex as re
import contractions
from textblob import TextBlob as tb

def remove_short_words(text):
        text = ' '.join(word for word in text.split() if len(word)>3)
        return text

def remove_num(text):
        text = re.sub('[0-9]+', '', text)
        return text

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def process_single_passage_production(passage_id, passages, stopwords, tb, stemmer, lemmatizer=None):
        # create a textblob containing the words of current passage
        curr_passage = passages[passage_id]
        curr_passage = remove_urls(curr_passage)
        curr_passage = remove_html(curr_passage)
        curr_passage = remove_num(curr_passage)

        text_blob = tb(curr_passage).words

        # lower words and remove stopwords from text blob
        no_stop_passages = [stemmer.stem(word.lower()) for word in text_blob if not word in stopwords]  #  
        return passage_id, no_stop_passages

def process_single_passage(passage_id, passages, stopwords, tb, stemmer, lemmatizer=None):
        # create a textblob containing the words of current passage
        curr_passage = passages[passage_id]

        text_blob = tb(curr_passage).words

        # lower words and remove stopwords from text blob
        no_stop_passages = [stemmer.stem(word.lower()) for word in text_blob if not word in stopwords]  #  
        return passage_id, no_stop_passages

def scrum_words(text):
    text=re.sub("(<.*?>)","",text)
    
    #remove non-ascii and digits
    text=re.sub("(\\W)"," ",text)

    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)
    
    #remove whitespace
    text=text.lower().strip()

    return text

def process_single_passage_experimental(passage_id, passages, stopwords, tb, stemmer=None):
        # retrieve sentence from passage dict
        sentence = passages[passage_id]
        # create new sentence with contractions extended, symbols removed and stopwords removed
        sen = " ".join([scrum_words(word) for word in sentence.split() if not scrum_words(word) in stopwords])
        result = [stemmer.stem(word) for word in sen.split() if len(word) > 1]
        # store the preprocessed result
        return passage_id, result

# def process_single_passage(passage_id, passages, stopwords, tb, stemmer=None, lemmatizer=None):
#         # create a textblob containing the words of current passage
#         curr_passage = passages[passage_id]

#         text_blob = tb(curr_passage).words

#         # lower words and remove stopwords from text blob
#         no_stop_passages = " ".join([word.lower() for word in text_blob if not word in stopwords])  # 
#         sentence = lemmatizer(no_stop_passages)
#         result = [token.lemma_ for token in sentence] 
#         return passage_id, result