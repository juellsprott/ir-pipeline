from gensim.models import Word2Vec

from pathlib import Path
import gensim.downloader as api
import numpy as np
import torch
from process import process_single_passage
from sklearn.metrics.pairwise import cosine_similarity
#from sentence_transformers import SentenceTransformer


def get_awe_similarities(
    qids,
    queries: dict[str, str],
    pids,
    passages: dict[str, str],
    stopwords,
    stemmer,
    lemmatizer=None,
):
    model_path = Path("./word2vec.model")
    if not model_path.is_file():
        # need to use model with dimensionality of 50
        model = api.load("glove-twitter-50")
        model.save("word2vec.model")

    else:
        model = Word2Vec.load("word2vec.model")

    vecs = dict()

    for qid, pid in zip(qids, pids):
        _, passage = process_single_passage(
            pid, passages, stopwords, stemmer, lemmatizer=lemmatizer
        )

        _, query = process_single_passage(
            qid, queries, stopwords, stemmer, lemmatizer=lemmatizer
        )

        passage_vecs = []
        for token in passage:
            if token in model.wv:
                passage_vecs += model.wv[token]

        query_vecs = []
        for token in query:
            if token in model.wv:
                query_vecs += model.wv[token]

        if len(passage_vecs) == 0 or len(query_vecs) == 0:
            sim = np.nan
        else:
            awe_p = sum(passage_vecs) / len(passage_vecs)
            awe_q = sum(query_vecs) / len(query_vecs)
            sim = cosine_similarity(awe_q, awe_p)

        vecs[qid][pid] = sim

    return vecs


def get_mpnet_similarities(
    qids,
    queries: dict[str, str],
    pids,
    passages: dict[str, str],
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    model = SentenceTransformer(model_name, device=device)

    vecs = dict()

    unique_qids = list(set(qids))
    unique_pids = list(set(pids))

    q_emb = model.encode([queries[q] for q in unique_qids], show_progress_bar=True)
    p_emb = model.encode([passages[p] for p in unique_pids], show_progress_bar=True)

    # create embedding lookups
    query_embeddings = dict()
    passage_embeddings = dict()

    for q_id, q_emb in zip(unique_qids, q_emb):
        query_embeddings[q_id] = q_emb

    for p_id, p_emb in zip(unique_pids, p_emb):
        passage_embeddings[p_id] = p_emb

    for q_id, p_id in zip(qids, pids):
        vecs[q_id][p_id] = cosine_similarity(
            query_embeddings[q_id], passage_embeddings[p_id]
        )

    return vecs
