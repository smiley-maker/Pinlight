import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import implicit

class Recommender:
    def __init__(self, item_embeddings, user_profiles, items_df, interactions_df, users_df):
        self.item_embeddings = item_embeddings
#        self.item_embeddings.drop(columns=[self.item_embeddings.columns[0], self.item_embeddings.columns[-1]], inplace=True)
        self.user_profiles = user_profiles
        self.items_df = items_df
        self.interactions = interactions_df
        self.users = users_df
        # Initialize the ALS model from implicit library
        self.model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.01, alpha=40, iterations=20)
        self.get_user_item_matrix()
        self.fit_model()

    def recommend_items(self, user_id, top_n=5):
        """
        Recommend items based on user profile similarity.
        
        Parameters:
        - user_id: ID of the user for whom to recommend items.
        - top_n: Number of top recommendations to return.
        
        Returns:
        - DataFrame of recommended items.
        """
        user_profile = self.user_profiles.loc[user_id]
        item_embeddings = self.item_embeddings.values
#        item_embeddings = self.item_embeddings.iloc[:, 1:].values
        
        # Calculate cosine similarity between user profile and item embeddings
        user_profile = user_profile.values.reshape(1, -1)  # Reshape to 2D array for cosine similarity
        similarities = cosine_similarity(user_profile, item_embeddings).flatten()

        # Remove the user's saved items from recommendations
        # Get saved items based on interactions data
        # This gives a list of indices of saved items
        saved_items = self.interactions[self.interactions['user_id'] == user_id]['item_id'].values
        # Remove saved items from the similarity scores
        similarities[saved_items] = -np.inf  # Set saved items' similarity to -inf to exclude them

        # Get indices of top N similar items
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        # Return recommended items from the actual fashion_items dataframe
        return self.items_df.iloc[top_indices]

    def get_user_item_matrix(self):
        user_item_matrix = [[0]*len(self.items_df) for _ in range(len(self.users))] # Initialize a 2D list with zeros
        for _, row in self.interactions.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            liked = row['liked']
            user_item_matrix[user_id][item_id] = liked # Set to 1 if liked, 0 otherwise

        self.user_item_matrix = csr_matrix(user_item_matrix) # Convert to sparse matrix        

    def fit_model(self):
        self.model.fit(self.user_item_matrix)

    def hybrid_filter(self, user_id, top_n=5, alpha=0.5):
        """
        Hybrid recommendation combining content-based and collaborative filtering.
        
        Parameters:
        - user_id: ID of the user for whom to recommend items.
        - top_n: Number of top recommendations to return.
        - alpha: Weighting factor between content-based and collaborative scores (0 <= alpha <= 1).
        
        Returns:
        - DataFrame of recommended items.
        """
        # Get content-based recommendations
        content_recs = self.recommend_items(user_id, top_n=top_n*2)  # Get more to allow for filtering later
        content_scores = np.ones(len(content_recs))  # Placeholder scores for content-based (could be improved)
        
        # Get collaborative filtering recommendations
        collab_recs = self.model.recommend(
            user_id,
            self.user_item_matrix[user_id],
            N=top_n*2,
            filter_already_liked_items=True
        )
        #print(collab_recs)
        collab_item_ids = [item_id for item_id, _ in zip(*collab_recs)]
        collab_scores = [score for _, score in zip(*collab_recs)]
    #    collab_item_ids, collab_scores = zip(*collab_recs)
        
        # Combine scores
        combined_scores = {}
        
        for idx, item in enumerate(content_recs['item_id']):
            combined_scores[item] = alpha * content_scores[idx]
        
        for idx, item in enumerate(collab_item_ids):
            if item in combined_scores:
                combined_scores[item] += (1 - alpha) * collab_scores[idx]
            else:
                combined_scores[item] = (1 - alpha) * collab_scores[idx]
        
        # Sort by combined score and get top N
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        recommended_item_ids = [item[0] for item in sorted_items]
        
        return self.items_df.loc[self.items_df['item_id'].isin(recommended_item_ids)]