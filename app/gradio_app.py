import sys
import os
print("CWD:", os.getcwd())

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import gradio as gr
import pandas as pd
from src.data_loader import DataLoader
from src.embedder import Embedder
from src.recommender import Recommender

# Load everything
loader = DataLoader()
users, items, interactions, item_embeddings, user_profiles = loader.load_data()
print(f"Loaded {len(users)} users, {len(items)} items, {len(interactions)} interactions.")
item_embeddings = item_embeddings.iloc[:, 1:]
print(f"Item embeddings has shape: {item_embeddings.shape}")
print(f"User profiles has shape: {user_profiles.shape}")

# Convert embeddings to match shape if needed
#user_profiles = {int(row['user_id']): row[1:].values for _, row in user_profiles.iterrows()}

# Init recommender
recommender = Recommender(
    item_embeddings=item_embeddings,
    user_profiles=user_profiles,
    items_df=items,
    interactions_df=interactions,
    users_df=users
)

def get_recommendations(user_id, alpha=0.5, top_n=5):
    user_id = int(user_id)

    # Interests
    interests = users.loc[users['user_id'] == user_id, 'interests'].values[0]

    # Previously liked pins
    liked_item_ids = interactions[interactions['user_id'] == user_id]['item_id'].unique()
    liked_items = items[items['item_id'].isin(liked_item_ids)][['title', 'tags']].reset_index(drop=True)

    # Recommended items
    recs = recommender.hybrid_filter(user_id, top_n=top_n, alpha=alpha)[['title', 'tags']].reset_index(drop=True)

    return interests, liked_items, recs



# Gradio interface
demo = gr.Interface(
    fn=get_recommendations,
    inputs=[
        gr.Dropdown(choices=users['user_id'].tolist(), label="Select User ID"),
        gr.Slider(0, 1, value=0.5, step=0.1, label="Hybrid Balance (alpha)"),
        gr.Slider(1, 10, value=5, step=1, label="Top N Recommendations"),
    ],
    outputs=[
        gr.Textbox(label="User Interests"),
        gr.Dataframe(label="Previously Liked Pins"),
        gr.Dataframe(label="Recommended Items"),
    ],
    title="🎽 Pinlight Fashion Recommender",
    description="Select a user to see their interests, liked pins, and new recommendations using a hybrid filtering model.",
    theme='soft'
)

if __name__ == "__main__":
    demo.launch()
