import pandas as pd
import numpy as np
import ast

class DataLoader:
    def __init__(self):
        self.users_path = "./data/fashion_users.csv"
        self.items_path = "./data/fashion_items.csv"
        self.interactions_path = "./data/fashion_interactions.csv"

    def load_data(self):
        items = pd.read_csv(self.items_path)
        # Convert string representations of tags list back to actual list
        items['tags'] = items['tags'].apply(ast.literal_eval)

        users = pd.read_csv(self.users_path)
        # Convert string representations of interests list back to actual list
        users['interests'] = users['interests'].apply(ast.literal_eval)

        interactions = pd.read_csv(self.interactions_path)

        item_embeddings = pd.read_csv("./data/item_embeddings.csv")
        user_profiles = pd.read_csv("./data/user_profiles.csv")

        return users, items, interactions, item_embeddings, user_profiles