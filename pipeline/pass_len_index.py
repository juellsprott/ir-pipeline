# computational processes
import multiprocessing
from functools import partial
from process import process_single_passage, process_single_passage_experimental
import pickle
from tqdm import tqdm
from textblob import TextBlob as tb
import os.path


class PassageLengthIndex(object):
    def __init__(self, data):
        self.data = data

    def chunk_passages(self, it, size):
        # create empty dict for chunks
        self.chunk = {}
        self.ctr = size
        # loop through passages to create chunks
        for key, val in it.items():
            self.chunk[key] = val
            self.ctr -= 1
            if self.ctr == 0:
                # yield dict chunk for preprocessing
                yield self.chunk
                self.ctr = size
                self.chunk = {}
        if self.chunk:
            yield self.chunk


    def create_pass_len_index(self, stopwords, pickle_name, tokenised_pass=None, stemmer=None, lemmatizer=None, result=True):
        """Preprocesses all passages"""
        
        print("Start preprocessing the passages.")

        self.index = {}
        self.name = pickle_name
        i = 0
        # check if a dict with tokenised passages has been passed to the function, if not, tokenize them here
        if tokenised_pass is None:
            # create a process pool that uses all cpus
            for passage_chunk in self.chunk_passages(self.data.passages, 500000):
                i += 1
                print(f"chunk: {i}")
                with multiprocessing.Pool() as pool:
                    # map the results to a list
                    results = pool.map(partial(process_single_passage_experimental, passages=passage_chunk, stopwords=stopwords, tb=tb, stemmer=stemmer, lemmatizer=lemmatizer), tqdm(passage_chunk.keys()))

                # convert preprocessed passage chunk to dictionary with pid: len(passage[pid])
                temp_dict = {k: len(v) for k, v in dict(results).items()}
                # update main index with new chunk
                self.index.update(temp_dict)
        else:
            # create passage len index using given tokenised passages
            self.index = {k: len(v) for k, v in tokenised_pass.items()}
        # store created index on disk
        with open(f"{self.name}_pass_len_index.pickle", 'wb') as f:
            pickle.dump(self.index, f)
        
        if result is True:
            return self.index


    def load_pass_len_index(self, pickle_name):
        # open the file path
        print("Trying to load index from disk...")
        if os.path.isfile(f"{pickle_name}_pass_len_index.pickle"):
            with open(f"{pickle_name}_pass_len_index.pickle", 'rb') as f:
                # load the index from the pickle file
                self.stored_index = pickle.load(f)
            print("Successfully loaded index to memory!")
            return self.stored_index
        else:
            print("Failed to load, because no stored index is found. Will now create a new index.")
            raise FileNotFoundError()
