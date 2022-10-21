import pandas as pd
from xgboost import XGBRanker


def transform_dict(
    train_dict, features
    # train_dict: dict[dict[str, float]], features: list[str]
) -> pd.DataFrame:
    """
    Takes in a scores[q_id][p_id] = [feat_1,...,feat_n] like 
    dict and transforms it into a pd.DataFrame usable for reranking
    """
    print("attempting to convert dict to df")
    train_df = pd.DataFrame.from_dict(train_dict)
    # current shape:
    #       q_id1   q_id2
    # p_id1 x       x
    # p_id2 x       x
    print("df created")
    train_df = train_df.stack().reset_index()
    train_df.columns = ["p_id", "q_id", "features"]
    #   p_id  q_id   features
    # 0 p_id1 q_id1   [x, y]
    # 1 p_id1 q_id2   [x, y]
    # 2 p_id2 q_id1   [x, y]
    # 3 p_id2 q_id2   [x, y]

    # features are now in a list
    # we want to give each feature their own column
    features_cols = train_df.pop("features")
    for feature, col in zip(features, zip(*list(features_cols.values))):
        train_df[feature] = col

    # final shape:
    #   p_id  q_id  feat1   feat2
    # 0 p_id1 q_id1 x       y
    # 1 p_id1 q_id2 x       y
    # 2 p_id2 q_id1 x       y
    # 3 p_id2 q_id2 x       y
    return train_df

def transform_dict_experimental(train_dict, columns) -> pd.DataFrame:
    train_df = pd.DataFrame([[q_id] + [p_id] + train_dict[q_id][p_id] for q_id in train_dict.keys() for p_id in train_dict[q_id].keys()])
    train_df.columns = columns
    return train_df

class XGBoostRanker:
    def __init__(self, objective="rank:map"):
        self.objective = objective
        self.model = None

    def fit_model(self, train_df, scores_names, label_name):
    # def fit(self, train_dict: dict[dict[str, float]]):
        self.model = XGBRanker(objective=self.objective, booster='dart')
        self.model.fit(
            train_df[scores_names],
            train_df[label_name],
            group=train_df.groupby("q_id").size().values,
        )

    def predict(self, test_df, scores_names):
        test_df = test_df[scores_names]
        print("predicting...")
        return self.model.predict(test_df[scores_names])


# test = {
#     "q_id1": {"p_id1": [25.0, 10.5], "p_id2": [22.0, 13.34]},
#     "q_id2": {"p_id1": [15.0, 10.2], "p_id2": [24.0, 0.23]},
# }

# print(transform_dict(test, features=["score", "tf_idf"]))
