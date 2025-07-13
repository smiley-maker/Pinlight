import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.nn.functional import normalize

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_item(self, item):
        title_embedding = self.model.encode(item['title'], convert_to_tensor=True)
        tags_embedding = self.model.encode(' '.join(item['tags']), convert_to_tensor=True)
        category_embedding = self.model.encode(item['category'], convert_to_tensor=True)

        combined = (0.3 * title_embedding + 0.5 * tags_embedding + 0.2 * category_embedding) / 3
        return normalize(combined.unsqueeze(0), p=2, dim=1).squeeze(0)

    def average_embeddings(self, user_id, embeddings_df, user_data):
        """
        Given a user ID, return the average of the embeddings for that user.

        Args:
            user_id (int): The ID of the user.
            embeddings_df (pd.DataFrame): DataFrame where the first column is item IDs and the
                                           subsequent columns are embedding dimensions.
            user_data (pd.DataFrame): DataFrame containing user interactions with columns
                                      'user_id' and 'item_id'.
        """
        # Obtain the user interactions for the specified user ID
        user_interactions = user_data[user_data['user_id'] == user_id]

        # Now get the item IDs that the user has interacted with
        item_ids = user_interactions['item_id'].unique()

        # Filter the DataFrame for the user's items
        user_items = embeddings_df.loc[item_ids]
        
        # If the user has no items, return None
        if user_items.empty:
            # Maybe we should return a zero vector instead?
            # return np.zeros(embeddings_df.shape[1] - 1) 
            return None
        
        # Calculate the average embedding vector
        avg_embedding = user_items.iloc[:, 1:].mean(axis=0)
        
        return avg_embedding
    
    def embed_user_profile(self, user_id, user_profiles, embeddings_df):
        avg_embedding = self.average_embeddings(user_id, embeddings_df)
        if avg_embedding is not None:
            user_profiles[int(user_id)] = avg_embedding.values
        else:
            user_profiles[int(user_id)] = np.zeros(embeddings_df.shape[1] - 1)
        
        return user_profiles

    def save_embeddings(self, embeddings, filepath):
        pd.to_csv(embeddings, filepath, header=False, index=False)

    def load_embeddings(self, filepath):
        return pd.read_csv(filepath, header=None)