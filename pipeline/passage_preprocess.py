# computational processes
import multiprocessing
from functools import partial
from process import process_single_passage_production
from process import process_single_passage, process_single_passage_experimental
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm
from textblob import TextBlob as tb
import os.path

class PassagePP(object):
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

    def process_passages(self, stopwords, stemmer):
        """Preprocesses all passages"""
        print("Start preprocessing the passages.")
        # create a process pool that uses all cpus
        self.index = {}
        i = 0
        # create a process pool that uses all cpus
        for passage_chunk in self.chunk_passages(self.data.passages, 1000000):
            i += 1
            print(f"chunk: {i}")
            with multiprocessing.Pool() as pool:
                # map the results to a list
                results = pool.map(partial(process_single_passage_production, passages=passage_chunk, stopwords=stopwords, tb=tb, stemmer=stemmer), tqdm(passage_chunk.keys()))

            # convert preprocessed passage chunk to dictionary with pid: len(passage[pid])
            temp_dict = dict(results)
            # update main index with new chunk
            self.index.update(temp_dict)
        return self.index