from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Initialize the Flask app
app = Flask(__name__)
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')  # Serve the index.html page from the current directory

# Load your dataset and preprocess it
dataset_path = "cleaned_dataset.csv"  # Update the path
df = pd.read_csv(dataset_path)
# Add data cleaning and TF-IDF processing here (same as in your notebook)

tfidf_overview = TfidfVectorizer(stop_words= "english")
df['overview'] = df['overview'].fillna("")
tfidx_matrix_overview = tfidf_overview.fit_transform(df['overview'])

tfidf_title = TfidfVectorizer(stop_words= "english")
df['title'] = df['title'].fillna("")
tfidx_matrix_title = tfidf_title.fit_transform(df['title'])

tfidf_genre = TfidfVectorizer(stop_words= "english")
df['genres'] = df['genres'].fillna("")
tfidx_matrix_genre = tfidf_genre.fit_transform(df['genres'])

tfidf_comp = TfidfVectorizer(stop_words= "english")
df['production_companies'] = df['production_companies'].fillna("")
tfidx_matrix_comp = tfidf_comp.fit_transform(df['production_companies'])

tfidf_key = TfidfVectorizer(stop_words= "english")
df['keywords'] = df['keywords'].fillna("")
tfidx_matrix_key = tfidf_key.fit_transform(df['keywords'])

tfidf_tag = TfidfVectorizer(stop_words= "english")
df['tagline'] = df['tagline'].fillna("")
tfidx_matrix_tag = tfidf_tag.fit_transform(df['tagline'])

similarities_overview = linear_kernel(tfidx_matrix_overview, tfidx_matrix_overview)

similarities_title = linear_kernel(tfidx_matrix_title, tfidx_matrix_title)

similarities_genre = linear_kernel(tfidx_matrix_genre, tfidx_matrix_genre)

similarities_comp = linear_kernel(tfidx_matrix_comp, tfidx_matrix_comp)

similarities_key = linear_kernel(tfidx_matrix_key, tfidx_matrix_key)

similarities_tag = linear_kernel(tfidx_matrix_tag, tfidx_matrix_tag)

import numpy as np


# Define weights for each similarity matrix
weight_overview = 0.3
weight_title = 0.1
weight_genre = 0.2
weight_comp = 0.2
weight_key = 0.1
weight_tag = 0.1

# Optional: Normalize the matrices if needed
similarities_overview = similarities_overview / np.max(similarities_overview)
similarities_title = similarities_title / np.max(similarities_title)
similarities_genre = similarities_genre / np.max(similarities_genre)
similarities_comp = similarities_comp / np.max(similarities_comp)
similarities_key = similarities_key / np.max(similarities_key)
similarities_tag = similarities_tag / np.max(similarities_tag)

# Combine similarities using a weighted sum
combined_similarities = (
    weight_overview * similarities_overview +
    weight_title * similarities_title +
    weight_genre * similarities_genre +
    weight_comp * similarities_comp +
    weight_key * similarities_key +
    weight_tag * similarities_tag
)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    title = data.get("title", "").strip()  # Extract title from JSON request

    # Get the index of the target movie
    if title in indices.index:
        # Find all matching indices
        matching_indices = indices[indices.index == title].tolist()

        # If there's more than one match, return all matches to the user
        if len(matching_indices) > 1:
            matches = []
            for movie_idx in matching_indices:
                movie_details = df.loc[movie_idx, ['title', 'release_date', 'vote_average']]
                matches.append({
                    "title": movie_details['title'],
                    "release_date": movie_details['release_date'],
                    "vote_average": movie_details['vote_average']
                })
            return jsonify({"matches": matches})

        # Proceed if only one matching title
        idx = matching_indices[0]
    else:
        return jsonify({"error": f"Movie with title '{title}' not found."})

    # Retrieve the similarity scores for the movie
    sim_score = list(enumerate(combined_similarities[idx]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    
    target_genres = set(df['genres'].iloc[idx].split(','))

    recommendations = []
    for i, score in sim_score[1:]:
        movie_genres = set(df['genres'].iloc[i].split(','))
        if target_genres & movie_genres:
            recommendations.append({
                "title": df['title'].iloc[i],
                "release_date": df['release_date'].iloc[i]
            })
        if len(recommendations) >= 10:
            break

    return jsonify({
        "title": df['title'].iloc[idx],
        "release_date": df['release_date'].iloc[idx],
        "recommendations": recommendations
    })

df_temp = df.copy()

#_temp Ensure 'keywords' and 'genres' columns have no missing values
df_temp.loc[:, 'keywords'] = df_temp['keywords'].fillna('')
df_temp.loc[:, 'genres'] = df_temp['genres'].fillna('')
@app.route('/recommend_by_genre', methods=['POST'])
def recommend_by_genre():
    data = request.get_json()
    title = data.get("title", "").strip()
    genre = data.get("genre", "").strip()

    if title in indices.index:
        matching_indices = indices[indices.index == title].tolist()
        if len(matching_indices) > 1:
            matches = []
            for movie_idx in matching_indices:
                movie_details = df.loc[movie_idx, ['title', 'release_date', 'vote_average']]
                matches.append({
                    "title": movie_details['title'],
                    "release_date": movie_details['release_date'],
                    "vote_average": movie_details['vote_average']
                })
            return jsonify({"matches": matches})

        idx = matching_indices[0]
    else:
        return jsonify({"error": f"Movie with title '{title}' not found."})

    sim_score = list(enumerate(combined_similarities[idx]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    
    target_keywords = set(df_temp['keywords'].iloc[idx].split(','))

    recommendations = []
    for i, score in sim_score[1:]:
        movie_keywords = set(df_temp['keywords'].iloc[i].split(','))
        movie_genres = set(df_temp['genres'].iloc[i].split(','))
        if target_keywords & movie_keywords and genre in movie_genres:
            recommendations.append({
                "title": df_temp['title'].iloc[i],
                "release_date": df_temp['release_date'].iloc[i]
            })
        if len(recommendations) >= 10:
            break

    return jsonify({
        "title": df_temp['title'].iloc[idx],
        "release_date": df_temp['release_date'].iloc[idx],
        "recommendations": recommendations
    })
if __name__ == '__main__':
    app.run(debug=True)