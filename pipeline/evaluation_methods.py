import numpy as np

class Precision:
    def __init__(self, queries, labels):
        self.queries = queries
        self.labels = labels
        
    def precision(self, scores: np.ndarray, k: int) -> float:
    
        retrieved = scores[:k]
        
        relevant_count = np.count_nonzero(retrieved)  # or np.sum
        retrieved_count = len(retrieved)
        
        if not relevant_count:
            return 0.0
        
        precision = relevant_count / retrieved_count
        
        return precision
    
    def evaluate(self, scores_dict, k):
        # only consider the position of the first passage that hit the ground truth 
        assert len(scores_dict) == len(self.labels)
        
        mean_rank = 0
        
        for q_id, p_ids in scores_dict.items():
            
            # the rank scores don't matter only ranking matters
            relevancy = []
            # for a certain query we need to see relevancy label of each document
            for idx, p_id in enumerate(p_ids):
                # get the value from the labeled documents, zero if not even in labels
                value = self.labels[q_id].get(p_id, 0)
                relevancy.append(value)
                    
            # now we have a nice ranked list with labeled relevancy
            # we can evaluate the ranked list
            mean_rank += self.precision(np.array(relevancy), k) 

        return mean_rank / len(scores_dict)
    
class MAP:
    def __init__(self, queries, labels):
        self.queries = queries
        self.labels = labels
        
    def precision(self, scores: np.ndarray, k: int) -> float:
    
        retrieved = scores[:k]
        
        relevant_count = np.count_nonzero(retrieved)  # or np.sum
        retrieved_count = len(retrieved)
        
        if not relevant_count:
            return 0.0
        
        precision = relevant_count / retrieved_count
        
        return precision
    
    def recall(self, scores: np.ndarray, k: int) -> float:
        total_relevant = np.count_nonzero(scores)
        
        retrieved = scores[:k]
        relevant_retrieved = np.count_nonzero(retrieved)
        
        # check for any relevant document
        if not relevant_retrieved:
            return 0.0
        
        
        recall = relevant_retrieved / total_relevant
        
        return recall
    
    def average_precision_at_n(self, scores: np.ndarray, q_id) -> float:
    
        average_precision = 0
        
        # number of total relevant documents
        
        labels = self.labels[q_id]
        # print(q_id)
        # print(labels)
        
        ground_truth_positives = len(labels)  # np.count_nonzero(scores)
        # print(ground_truth_positives)

        # print("\n")
        if not ground_truth_positives:
            return 0.0
        
        # compute for every k the precision and recall
        
        # k is not zero indexed, but range(len()) is
        # for example a top k=20, we grab last index 19, but we give k=20 to precision
        for k in range(1, len(scores) + 1):
            # create boolean for the relevance at k (which is at k - 1, last item of scores[:k])
            relevance_at_k = (scores[k - 1] != 0)
            
            average_precision += (self.precision(scores, k) * relevance_at_k)
        
        # normalize by the total relevant documents
        average_precision *= (1 / ground_truth_positives)
            
        return average_precision
    
    def evaluate(self, scores_dict):
        # only consider the position of the first passage that hit the ground truth 
        assert len(scores_dict) == len(self.labels)
        
        summed_average_precision = 0
        
        for q_id, p_ids in scores_dict.items():
            
            # the rank scores don't matter only ranking matters
            relevancy = []
            # for a certain query we need to see relevancy label of each document
            for idx, p_id in enumerate(p_ids):
                # get the value from the labeled documents, zero if not even in labels
                value = self.labels[q_id].get(p_id, 0)
                relevancy.append(value)
                    
            # now we have a nice ranked list with labeled relevancy
            # we can evaluate the ranked list
            summed_average_precision += self.average_precision_at_n(np.array(relevancy), q_id) 

        mean_average_precision = summed_average_precision / len(scores_dict)
        return mean_average_precision
    
class MRR:
    def __init__(self, queries, labels):
        self.queries = queries
        self.labels = labels
    
    def evaluate(self, scores, k):
        # only consider the position of the first passage that hit the ground truth 
        assert len(scores) == len(self.labels)
        
        reciprocal_rank = 0
        
        for q_id, p_ids in scores.items():
            
            for index, p_id in enumerate(p_ids):
                if index == k:
                    break
                
                # if this doc is indeed a relevant doc for q
                if p_id in self.labels[q_id]:
                    reciprocal_rank += 1.0 / (index+1)
                    break 

        return reciprocal_rank / len(scores)
    
class nDCG:
    def __init__(self, queries, labels):
        self.queries = queries
        self.labels = labels
    
    def dcg(self, scores: np.ndarray, k: int) -> float:
        dcg = np.sum(scores[:k] / np.log2(np.indices(scores[:k].shape)[0] + 2))
        return dcg
    
    def ndcg(self, scores, k):
        ndcg = 0.0

        reverse_sort = np.array(sorted(scores, reverse=True))
        
        ideal = self.dcg(reverse_sort, k)
        
        if ideal == 0:
            return 0
        
        real = self.dcg(scores, k)
        
        ndcg = real / ideal
        
        return ndcg

    def evaluate(self, scores_dict, k):
        # only consider the position of the first passage that hit the ground truth 
        assert len(scores_dict) == len(self.labels)
        
        ndcg_rank = 0
       
        # for every query
        for q_id, p_ids in scores_dict.items():
            # the rank scores don't matter only ranking matters
            relevancy = []
            # for a certain query we need to see relevancy label of each document
            for idx, p_id in enumerate(p_ids):
                # get the value from the labeled documents, zero if not even in labels
                value = self.labels[q_id].get(p_id, 0)
                relevancy.append(value)
                    
            # now we have a nice ranked list with labeled relevancy
            # we can evaluate the ranked list
            ndcg_rank += self.ndcg(np.array(relevancy), k) 
            

        return ndcg_rank / len(scores_dict)
    
    
class DCG:
    def __init__(self, queries, labels):
        self.queries = queries
        self.labels = labels
    
    def dcg(self, scores: np.ndarray, k: int) -> float:
        dcg = np.sum(scores[:k] / np.log2(np.indices(scores[:k].shape)[0] + 2))
        return dcg

    def evaluate(self, scores_dict, k):
        # only consider the position of the first passage that hit the ground truth 
        assert len(scores_dict) == len(self.labels)
        
        dcg_rank = 0
       
        # for every query, with predicted documents (list)
        for q_id, p_ids in scores_dict.items():
            # the rank scores don't matter only ranking matters
            relevancy = []
            # for a certain query we need to see relevancy label of each document
            for idx, p_id in enumerate(p_ids):
                # get the value from the labeled documents, zero if not even in labels
                value = self.labels[q_id].get(p_id, 0)
                relevancy.append(value)
                    
            # now we have a nice ranked list with labeled relevancy
            # we can evaluate the ranked list
            dcg_rank += self.dcg(np.array(relevancy), k) 
            

        return dcg_rank / len(scores_dict)