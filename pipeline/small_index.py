# computational processes
import multiprocessing
from functools import partial
from process import process_single_passage_experimental, process_single_passage
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm
from textblob import TextBlob as tb
import os.path

class SmallIndex(object):
    def __init__(self, data):
        self.data = data

    def process_passages(self, passages, stopwords, stemmer, lemmatizer=None, return_results=False):
        """Preprocesses all passages"""
        print("Start preprocessing the passages.")
        # create a process pool that uses all cpus
        with multiprocessing.Pool() as pool:
            # map the results to a list
            results = pool.map(partial(process_single_passage, passages=passages, stopwords=stopwords, tb=tb, stemmer=stemmer, lemmatizer=lemmatizer), tqdm(passages.keys()))

        # dict with {passage_id : preprocessed_text}
        print("Store preprocessed passages.")
        self.data.tokenised_passages = dict(results)
        if return_results == True:
            return self.data.tokenised_passages


        
    def create_index(self, pickle_name, stopwords, stemmer, tokenised_pass=None, lemmatizer=None, result=True):
        assert self.data.passages is not None, """It seems like the data has no passages. Please assign
        passages to the data, before creating SmallIndex instance"""
        if tokenised_pass is None:
            self.process_passages(self.data.passages, stopwords, stemmer, lemmatizer)
            self.passages = self.data.tokenised_passages
        else:
            self.passages = tokenised_pass

        self.name = pickle_name
        
        
        self.num_p = len(self.passages)
        
        # intialize dummy index with duplicate passage_ids
        self.duplicate_index = {}

        # initialize index for holding lengths of each passage
        self.passage_lengths = {}

        # create a final index with correct format
        self.index = defaultdict(dict)

        total_tokens = 0
        
        # create dummy index for all tokens from the passages
        for passage_id, tokens in tqdm(self.passages.items()):
            total_tokens += len(tokens)
            self.passage_lengths[passage_id] = len(tokens)
            for token in tokens:
                
                # if the token from the passage already has a posting
                if token in self.duplicate_index.keys():
                    
                    # append the passage to the posting
                    self.duplicate_index[token].append(passage_id)
                else:
                    # or create a new posting with this passage
                    self.duplicate_index[token] = [passage_id]

        # store the number of passages
        self.index['total_documents'] = self.num_p

        print("Create the index.")
        # for every token and its dummy postings
        for token, dummy_posting in tqdm(self.duplicate_index.items()):
            # convert the dummy postings to postings with frequencies
            frequencies = Counter(dummy_posting)
            
            postings = []
            # convert Counter(passage_id:freq) to list of dicts
            for passage_id, freq in frequencies.items():
                postings.append({'id': passage_id,'term_frequency': freq, 'term_prob': freq / self.passage_lengths[passage_id]})
                
            # document frequency is the amount of documents containing the token
            doc_freq = len(postings)
            corp_freq = len(dummy_posting) / total_tokens
            # add those stats to the token stats
            all_passages = list(map(lambda d: d['id'], postings))
            self.index[token]['corpus_frequency'] = corp_freq
            self.index[token]['postings'] = postings
            self.index[token]['document_frequency'] = doc_freq
            self.index[token]['passage_list'] = all_passages
        

        print("Create and store index.")
        # store the index in a pickle
        with open(f"{self.name}_index.pickle", 'wb') as f:
            pickle.dump(self.index, f)

        if result:
            return self.index

    def load_index(self, pickle_name):
        name = f"{pickle_name}_index.pickle"
        # open the file path
        print("Trying to load index from disk...")
        if os.path.isfile(name):
            with open(name, 'rb') as f:
                # load the index from the pickle file
                self.stored_index = pickle.load(f)
            print("Successfully loaded index to memory!")
            return self.stored_index
        else:
            print("File not found! Check if you're in the correct directory or run create_index.")
            raise FileNotFoundError()