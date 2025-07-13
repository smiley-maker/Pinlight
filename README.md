# Pinlight

**A Pinterest‑style hybrid recommender system for fashion content**, built to explore cold-start personalization and behavior-driven discovery.

---

## Project Overview

**Pinlight** simulates a lightweight recommendation engine using:

* **Content-Based Filtering**: Leverages semantic embeddings (title, tags, category) to estimate user affinity — ideal for cold-start.
* **Collaborative Filtering**: Uses ALS (via the `implicit` library) to capture patterns from user-item interaction history.
* **Hybrid Model**: Combines both techniques with a tunable weighted approach.

A live **Gradio demo** allows users to:

* Select a user ID
* View their interests and past liked pins
* Explore top‑N recommendations
* Adjust the content/collaborative balance (`alpha`)

---

## Repository Structure

```
Pinlight/
├── app/                  ← Gradio demo interface
├── data/                 ← Simulated CSVs: users, items, interactions, embeddings
├── notebooks/            ← Exploration & model-development notebooks
├── src/
│   ├── data_loader.py    ← Loads and preprocesses CSVs
│   ├── embedder.py       ← Embeds items; builds user profiles
│   └── recommender.py    ← Implements CF, CBF, and hybrid recommendations
├── requirements.txt      ← Python dependencies
└── README.md
```

---

## Quickstart Instructions

1. **Create a virtual or conda environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   Or if using conda, 
   ```bash
   conda create -n pinlight python=3.12
   conda activate pinlight
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo app** from the repo root:

   ```bash
   python app/gradio_app.py
   ```

   Then open the URL that Gradio outputs (e.g., `http://localhost:7860/`) to explore the interactive recommender.

---

## Core Features

* ✨ **Cold-Start Friendly**: Content-based embeddings allow recommendations from a single interaction.
* 📊 **User-Centric**: Shows each user’s interests and previously liked pins.
* 🔄 **Hybrid Flexibility**: Adjustable `alpha` enables blending content and collaborative signals at different ratios.
* 📈 **Visual Exploration**: Gradio demo interface offers intuitive control and transparency.

---

## Evaluation & Insights

* Offline analysis shows hybrid models outperform pure content or collaborative approaches, balancing the tradeoff between new recommendations based on other user data and managing the cold-start problem with content-based recommendations.
* Sample user outputs reveal that even with sparse history, recommendations align well with inferred preferences.
* Demonstrates how content and user behavior can complement each other for richer personalization.

---

## Next Steps

* Add image support (show pin thumbnails).
* Allow custom interest-tag input (e.g., "I like streetwear") and new users.
* Build a meta-learner to dynamically optimize `alpha`.
* Scale to real-world datasets for deeper validation.