# import torch.utils.data as data
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import os
# import json
import collections
import codecs
from full_ranker import Fullranker
from tqdm import tqdm
from embedding_similarities import get_awe_similarities, get_mpnet_similarities


class VectorCreator(object):
    def __init__(self, index, pass_len_index, labels, queries, path):
        self.index = index
        self.labels = labels
        self.path = path
        self.pass_len_index = pass_len_index
        self.queries = queries
        self.params = {"alpha": 0.7, "b": 0.8, "k1": 0.7, "l": self.pass_len_index}
        self.franker = Fullranker(self.index)

    def get_tfidf(self, pids):
        print("obtain tf idf scores")
        scores = self.franker.tf_idf_score(self.queries, **self.params)

        return scores

    def get_QL(self, pids):
        print("obtain ql scores")
        scores = self.franker.query_likelihood(self.queries, **self.params)

        return scores

    def get_AWE(
        self, qids, queries, pids, passages, stopwords, stemmer, lemmatizer=None
    ):
        print("obtain AWE vectors")
        scores = get_awe_similarities(
            qids, queries, pids, passages, stopwords, stemmer, lemmatizer
        )

        return scores

    def get_MPNet(self, qids, queries, pids, passages):
        print("obtain MPNet vectors")
        scores = get_mpnet_similarities(qids, queries, pids, passages)

        return scores

    def get_vectors(self):
        # load the full-ranking result on the training set.
        q_ids = []
        p_ids = []
        features = collections.defaultdict(lambda: collections.defaultdict(list))

        # print("Load file {}".format("output/full_ranking_training_result.text"))

        MAIN_FEATURE = "bm25"

        # load main feature training results (BM25)
        with codecs.open(self.path, "r", "utf-8") as file:
            # read through the lines of the file from memory
            for line in file.readlines():
                # get every entry
                content = line.split("\t")  # list

                # store the query id
                q_id = content[0]
                p_id = content[1]

                q_ids.append(q_id)
                p_ids.append(p_id)

                main_feature_score = float(content[3])
                if self.labels is not None:
                    label = self.labels[q_id].get(p_id, 0)
                    features[q_id][p_id].extend([main_feature_score, label])
                else:
                    features[q_id][p_id].extend([main_feature_score])
                # features.append([float(content[3]),float(content[4])])

        # create score matrices containing only the relevant passages

        tfidf_matrix = self.get_tfidf(p_ids)
        QL_matrix = self.get_QL(p_ids)
        # AWE_matrix = self.get_AWE(self, q_ids, p_ids)
        print("score matrices obtained, creating vectors")

        # loop through all q_id - p_id pairings and get obtain their corresponding features
        for q_id, p_id in tqdm(zip(q_ids, p_ids)):
            query_term_count = len(self.queries[q_id])
            tfidf_score = tfidf_matrix[q_id][p_id]
            ql_score = QL_matrix[q_id][p_id]
            passage_length_count = self.pass_len_index[p_id]
            # awe = AWE_matrix[q_id][p_id]

            # extend current main feature with additional features
            features[q_id][p_id].extend(
                [tfidf_score, ql_score, query_term_count, passage_length_count]
                # [tfidf_score, ql_score, query_term_count, passage_length_count, awe]
            )

        return features

    def get_vectors_with_awe(
        self, queries, passages, stopwords, stemmer, lemmatizer=None
    ):
        # load the full-ranking result on the training set.
        q_ids = []
        p_ids = []
        features = collections.defaultdict(lambda: collections.defaultdict(list))

        MAIN_FEATURE = "bm25"

        # load main feature training results (BM25)
        with codecs.open(self.path, "r", "utf-8") as file:
            # read through the lines of the file from memory
            for line in file.readlines():
                # get every entry
                content = line.split("\t")  # list

                # store the query id
                q_id = content[0]
                p_id = content[1]

                q_ids.append(q_id)
                p_ids.append(p_id)

                main_feature_score = float(content[3])
                label = self.labels[q_id].get(p_id, 0)
                features[q_id][p_id].extend([main_feature_score, label])

        # create score matrices containing only the relevant passages
        tfidf_matrix = self.get_tfidf(p_ids)
        QL_matrix = self.get_QL(p_ids)
        AWE_matrix = self.get_AWE(
            self, q_ids, queries, p_ids, passages, stopwords, stemmer, lemmatizer
        )

        print("score matrices obtained, creating vectors")
        # loop through all q_id - p_id pairings and get obtain their corresponding features
        for q_id, p_id in tqdm(zip(q_ids, p_ids)):
            query_term_count = len(self.queries[q_id])
            tfidf_score = tfidf_matrix[q_id][p_id]
            ql_score = QL_matrix[q_id][p_id]
            passage_length_count = self.pass_len_index[p_id]
            awe_sim = AWE_matrix[q_id][p_id]

            # extend current main feature with additional features
            features[q_id][p_id].extend(
                [tfidf_score, ql_score, query_term_count, passage_length_count, awe_sim]
            )

        return features

    def get_vectors_with_mpnet(self, queries, passages):
        # load the full-ranking result on the training set.
        q_ids = []
        p_ids = []
        features = collections.defaultdict(lambda: collections.defaultdict(list))

        MAIN_FEATURE = "bm25"

        # load main feature training results (BM25)
        with codecs.open(self.path, "r", "utf-8") as file:
            # read through the lines of the file from memory
            for line in file.readlines():
                # get every entry
                content = line.split("\t")  # list

                # store the query id
                q_id = content[0]
                p_id = content[1]

                q_ids.append(q_id)
                p_ids.append(p_id)

                main_feature_score = float(content[3])
                label = self.labels[q_id].get(p_id, 0)
                features[q_id][p_id].extend([main_feature_score, label])

        # create score matrices containing only the relevant passages

        tfidf_matrix = self.get_tfidf(p_ids)
        QL_matrix = self.get_QL(p_ids)
        MPNet_matrix = self.get_MPNet(q_ids, queries, p_ids, passages)

        print("score matrices obtained, creating vectors")

        # loop through all q_id - p_id pairings and get obtain their corresponding features
        for q_id, p_id in tqdm(zip(q_ids, p_ids)):
            query_term_count = len(self.queries[q_id])
            tfidf_score = tfidf_matrix[q_id][p_id]
            ql_score = QL_matrix[q_id][p_id]
            passage_length_count = self.pass_len_index[p_id]
            mpnet_sim = MPNet_matrix[q_id][p_id]

            # extend current main feature with additional features
            features[q_id][p_id].extend(
                [
                    tfidf_score,
                    ql_score,
                    query_term_count,
                    passage_length_count,
                    mpnet_sim,
                ]
            )

        return features
