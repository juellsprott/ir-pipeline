from process import process_single_passage
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import itertools


class Fullranker(object):
    def __init__(self, index):
        self.index = index

    def tf_score(self, queries, **kwargs):
        num_q = len(queries)
        scores = defaultdict(dict)

        for q_id, q_tokens in tqdm(queries.items()):
            for q_token in q_tokens:
                if q_token not in self.index.keys():
                    continue

                for passage_dict in self.index[q_token]["postings"]:
                    p_id = passage_dict["id"]
                    tf = 1 + np.log(passage_dict["term_frequency"])

                    if p_id in scores[q_id]:
                        scores[q_id][p_id] += tf
                    else:
                        scores[q_id][p_id] = tf

        return scores

    def tf_idf_score(self, queries, **kwargs):
        num_q = len(queries)
        scores = defaultdict(dict)

        # for every query
        for q_id, q_tokens in tqdm(queries.items()):

            # for each token
            for q_token in q_tokens:
                if q_token not in self.index.keys():
                    continue

                idf = np.log(
                    self.index["total_documents"]
                    / self.index[q_token]["document_frequency"]
                )

                # iterate through postings
                for passage_dict in self.index[q_token]["postings"]:
                    p_id = passage_dict["id"]
                    tf = 1 + np.log(passage_dict["term_frequency"])

                    score = tf * idf
                    if p_id in scores[q_id]:
                        scores[q_id][p_id] += score
                    else:
                        scores[q_id][p_id] = score

        return scores


    def query_likelihood(self, queries, **kwargs):
        scores = defaultdict(dict)
        self.alpha = kwargs['alpha']
        self.queries = queries
        i = 0
        a_minus = (1-self.alpha)
        
        # for every query 
        for q_id, q_tokens in tqdm(self.queries.items()):
            i+=1

            prob_list = defaultdict(list)

            # for each token, calculate probability of token in passage using SQL formula
            for q_token in q_tokens:
                if q_token not in self.index.keys():
                    continue
                # grab probability of query token in corpus
                prob_corpus = self.alpha * self.index[q_token]['corpus_frequency']
                for postings in self.index[q_token]['postings']:
                    prob = postings['term_prob']
                    final_prob = prob_corpus + (a_minus * prob)
                    prob_list[postings['id']].append((q_token, final_prob))

            for q_token in q_tokens:
                
                if q_token not in self.index.keys():
                    continue

                corp_freq = self.index[q_token]['corpus_frequency']
                for k,v in prob_list.items():
                    tokens = [i[0] for i in v]
                    if q_token not in tokens:
                        val = self.alpha * corp_freq
                        prob_list[k].append((q_token, val))

            for p_id, probs in prob_list.items():
                scores[q_id][p_id] = np.prod(np.array([i[1] for i in probs]))

                    
        
        return scores 

    # def smoothed_query_likelihood(self, queries, alpha=0.1):
    #     num_q = len(queries)
    #     scores = defaultdict(dict)
    #     self.alpha = alpha
    #     summed_probs = defaultdict(list)
    #     a_minus = 1 - self.alpha

    #     # for every query
    #     for q_id, q_tokens in tqdm(queries.items()):
    #         in_index = 0

    #         temp = []
    #         prob_list = []

    #         for q_token in q_tokens:
    #             if q_token in self.index.keys():
    #                 temp.append(
    #                     list(map(lambda d: d["id"], self.index[q_token]["postings"]))
    #                 )

    #         viewed_passages = list(itertools.chain.from_iterable(temp))

    #         for p_id in tqdm(viewed_passages):

    #             # for each token, calculate probability of token in passage using SQL formula
    #             for q_token in q_tokens:
    #                 if q_token not in self.index.keys():
    #                     continue
    #                 # grab probability of query token in corpus
    #                 prob_corpus = self.alpha * self.index[q_token]["corpus_frequency"]
    #                 pass_list = self.index[q_token]["passage_list"]
    #                 try:
    #                     id = pass_list.index(p_id)
    #                     # prob = self.index[q_token]['postings'][id]['term_prob']
    #                     # final_prob = prob_corpus + (a_minus * prob)
    #                     # prob_list.append((p_id, final_prob))
    #                 except ValueError:
    #                     continue

    #         for p_id, probs in prob_list:
    #             if scores[q_id].get(p_id, 0) == 0:
    #                 scores[q_id][p_id] = probs
    #             else:
    #                 scores[q_id][p_id] = scores[q_id][p_id] * probs

    #     return scores

    def BM25_score(self, queries, **kwargs):
        #  def query_likelihood(index: Dict, tokens: List[str], top_k: int) -> List[Tuple[str, float]]:
        k_1 = kwargs['k1']
        b = kwargs['b']
        scores = defaultdict(dict)
        length_index = kwargs['l']
        #avgdl = sum(length_index.values()) / len(length_index.keys())

        # for every query
        for q_id, q_tokens in tqdm(queries.items()):
            # for each token
            for q_token in q_tokens:
                if q_token not in self.index.keys():
                    continue

                avgdl = np.average(
                    [
                        length_index[passage["id"]]
                        for passage in self.index[q_token]["postings"]
                    ]
                )
                curr_idf = np.log(
                    self.index["total_documents"]
                    / self.index[q_token]["document_frequency"]
                )
                for passage in self.index[q_token]["postings"]:
                    p_id = passage["id"]
                    p_tf = passage["term_frequency"]

                    curr_len = length_index[p_id]
                    curr_bm_25 = curr_idf * ((
                        (p_tf * (k_1 + 1))
                        / (p_tf + k_1 * (1 - b + b * curr_len / avgdl))
                    ))

                    if scores[q_id].get(p_id, 0) == 0:
                        scores[q_id][p_id] = curr_bm_25
                    else:
                        scores[q_id][p_id] += curr_bm_25

        return scores

    def pseudo_relevance_feedback(self, queries, top_k_passages, labels):

        self.new_queries = {}

        self.get_passage_tokens(top_k_passages)

        for q_id, q_tokens in queries.items():
            for q_token in q_tokens:
                if q_token not in self.index.keys():
                    continue

                idf = np.log(
                    self.index["total_documents"]
                    / self.index[q_token]["document_frequency"]
                )

                for passage in top_k_passages:
                    if passage in labels[q_id]:
                        pass

    def get_passage_tokens(self, top_100_passages):
        self.tokens = []
        for passage_id in tqdm(top_100_passages):
            for token in self.index.keys():
                if token != 'total_documents' and token != 'elexgo':
                    for posting in self.index[token]['postings']:
                        if passage_id == posting['id']:
                            self.tokens.append(token)
        return self.tokens
