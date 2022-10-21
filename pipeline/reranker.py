from xgboost_ranker import transform_dict_experimental
from process import process_single_passage
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import itertools
import codecs
import pandas as pd
from xgboost_ranker import transform_dict, XGBoostRanker
from ranknet import train, inference
import os



class Reranker(object):
    def __init__(self, features_training, features_validation, name, features_test=None, ranknet_args=None):
        self.features = features_training
        self.features_validation = features_validation
        self.features_test = features_test
        self.args = ranknet_args
        self.results_name = name

    def xgboost_reranker(self, test=False, awe=True, lsa=False):
        # get feature vector for training data
        column_names = ["bm25_score", "label", "tfidf_score", 
                    "ql_score", "query_term_count", "passage_length_count"]

        column_names_test = ["bm25_score", "tfidf_score", 
                    "ql_score", "query_term_count", "passage_length_count"]
                
        train_names = ["bm25_score", "tfidf_score", 
                    "ql_score", "query_term_count", "passage_length_count"]

        if awe == True:
            column_names.append("awe")
            column_names_test.append("awe")
            train_names.append("awe")
        if lsa == True:
            column_names.append("lsa")
            column_names_test.append("lsa")
            train_names.append("lsa")
        
        features_df = transform_dict_experimental(self.features, ["q_id", "p_id"] + column_names)
        # get feature vector for either test or validation data
        if test == False:
            features_df_validation = transform_dict_experimental(self.features_validation, ["q_id", "p_id"] + column_names)
            results_name = f"reranker_output/validation/{self.results_name}_xgboost_validation_result.text"
        elif test == True:
            features_df_validation = transform_dict_experimental(self.features_test, ["q_id", "p_id"] + column_names_test)
            results_name = f"reranker_output/test/{self.results_name}_xgboost_test_result.text"

        # create xgboost object
        print("Finished transforming dict to df, fitting...")
        xgbr = XGBoostRanker()

        # fit on training features
        xgbr.fit_model(features_df, train_names, "label")
        print("Fit successful, predicting...")

        # predict using test/validation features
        preds = xgbr.predict(features_df_validation, train_names)
        # attach predictions to feature vectors
        features_df_validation['xgboost'] = preds
        # sort feature df by xgboost results
        features_df_validation = features_df_validation.sort_values(by=['q_id', 'xgboost'], ascending=[True, False])
        
        # write results to file
        print("Predict successful, saving results...")

        ranking = 0
        with codecs.open(results_name, "w", "utf-8") as file:
            for index, row in features_df_validation.iterrows():
                if ranking == 100:
                    ranking = 0
                ranking+=1           
                
                file.write('\t'.join([row['q_id'], row['p_id'], str(ranking), str(row['xgboost']), "re_ranking_on_the_validation_set"])+os.linesep)

        # output the result file. 
        print("Produce file {}".format(results_name)) 
        return results_name

    def ranknet_training(self, column_names, train_names):
        # transform feature vector into pandas dataframe
        
        features_df = transform_dict_experimental(self.features, ["q_id", "p_id"] + column_names)
        

        # convert df columns to lists for training
        q_ids = features_df['q_id'].tolist()
        ranknet_features = features_df[train_names].values.tolist()
        labels = features_df['label'].tolist()
        print("start training")
        train(self.args, q_ids, ranknet_features, labels)



    def ranknet_reranker(self, test=False, awe=True, lsa=False):
        # train ranknet
        column_names = ["bm25_score", "label", "tfidf_score", 
                    "ql_score", "query_term_count", "passage_length_count"]

        column_names_test = ["bm25_score", "tfidf_score", 
                    "ql_score", "query_term_count", "passage_length_count"]

        train_names = ["bm25_score", "tfidf_score", 
                    "ql_score", "query_term_count", "passage_length_count"]

        if awe == True:
            column_names.append("awe")
            column_names_test.append("awe")
            train_names.append("awe")
        if lsa == True:
            column_names.append("lsa")
            column_names_test.append("lsa")
            train_names.append("lsa")

        if test==False:
            self.ranknet_training(column_names, train_names)

        print("Training complete, creating inference features")
        if test == False:
            features_df = transform_dict_experimental(self.features_validation, ["q_id", "p_id"] +  column_names)
            results_name = f"reranker_output/validation/{self.results_name}_ranknet_validation_result.text"
        elif test == True:
            features_df = transform_dict_experimental(self.features_test, ["q_id", "p_id"] + column_names_test)
            results_name = f"reranker_output/test/{self.results_name}_ranknet_test_result.text"
        
        # convert df columns to lists for inference
        q_ids = features_df['q_id'].tolist()
        
        ranknet_features = features_df[train_names].values.tolist()
        p_ids = features_df['p_id'].tolist()
        print("Feature vectors created, starting inference")
        scores = inference(self.args, q_ids, p_ids, ranknet_features)
        print("Predictions done, saving output to file...")
        for q_id, p2score in tqdm(scores.items()):
            sorted_p2score=sorted(p2score.items(), key=lambda x:x[1], reverse = True)
            scores[q_id]=sorted_p2score

        with codecs.open(results_name, "w", "utf-8") as file:
            for q_id, p2score in tqdm(scores.items()):
                ranking=0
                for (p_id, score) in p2score:
                    ranking+=1           
                            
                    file.write('\t'.join([q_id, p_id, str(ranking), str(score), "re_ranking_on_the_validation_set"])+os.linesep)

        # output the result file. 
        print("Produce file {}".format(results_name)) 
        return results_name
