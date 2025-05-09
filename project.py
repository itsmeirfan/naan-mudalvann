import pandas as pd
import numpy as np
import re
import os
import math
import zipfile
import urllib.request
from io import BytesIO
from collections import Counter, defaultdict

# ========== DATA LOADING ==========

def download_movielens_data():
    """Download MovieLens dataset"""
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Download dataset
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    print(f"Downloading dataset from {url}...")
    
    with urllib.request.urlopen(url) as response:
        with zipfile.ZipFile(BytesIO(response.read())) as zip_ref:
            zip_ref.extractall("data")
    
    # Data paths
    movies_path = "data/ml-latest-small/movies.csv"
    ratings_path = "data/ml-latest-small/ratings.csv"
    
    return movies_path, ratings_path

# ========== CUSTOM TF-IDF IMPLEMENTATION ==========

def compute_tfidf(documents):
    """Compute TF-IDF for a list of documents"""
    # Tokenize documents (split by space)
    doc_tokens = [doc.split() for doc in documents]
    
    # Calculate document frequency
    term_doc_freq = defaultdict(int)
    for tokens in doc_tokens:
        for term in set(tokens):
            term_doc_freq[term] += 1
    
    # Total document count
    n_docs = len(documents)
    
    # Compute TF-IDF vectors
    tfidf_vectors = []
    for tokens in doc_tokens:
        # Count term frequency
        term_freq = Counter(tokens)
        
        # Calculate TF-IDF vector
        vector = {}
        for term, freq in term_freq.items():
            # TF-IDF = term frequency * log(total docs / docs with term)
            idf = math.log(n_docs / (1 + term_doc_freq[term]))
            vector[term] = freq * idf
        
        tfidf_vectors.append(vector)
    
    return tfidf_vectors

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    # Find common terms
    common_terms = set(vec1.keys()) & set(vec2.keys())
    
    # If no common terms, similarity is 0
    if not common_terms:
        return 0
    
    # Calculate dot product
    dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
    
    # Calculate magnitudes
    mag1 = math.sqrt(sum(val**2 for val in vec1.values()))
    mag2 = math.sqrt(sum(val**2 for val in vec2.values()))
    
    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0
    
    return dot_product / (mag1 * mag2)

# ========== CONTENT-BASED FILTERING ==========

def content_based_recommendations(movies_df, movie_title, n=5):
    """Get movie recommendations based on genre similarity"""
    # Prepare genre data
    movies_df['genres_text'] = movies_df['genres'].str.replace('|', ' ')
    
    # Create TF-IDF vectors
    tfidf_vectors = compute_tfidf(movies_df['genres_text'].tolist())
    
    # Find the movie index
    movie_idx = None
    for i, title in enumerate(movies_df['title']):
        if title.lower() == movie_title.lower():
            movie_idx = i
            break
    
    # If not found, try finding a partial match
    if movie_idx is None:
        for i, title in enumerate(movies_df['title']):
            if movie_title.lower() in title.lower():
                movie_idx = i
                print(f"Using closest match: {title}")
                break
    
    # If still not found
    if movie_idx is None:
        print(f"Movie '{movie_title}' not found")
        return pd.DataFrame()
    
    # Calculate similarities
    similarities = [(i, cosine_similarity(tfidf_vectors[movie_idx], vec)) 
                    for i, vec in enumerate(tfidf_vectors)]
    
    # Sort by similarity (excluding the input movie)
    similarities = [(i, score) for i, score in similarities if i != movie_idx]
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    top_indices = [i for i, score in similarities[:n]]
    recommendations = movies_df.iloc[top_indices].copy()
    recommendations['similarity'] = [score for _, score in similarities[:n]]
    
    return recommendations[['title', 'genres', 'similarity']]

# ========== COLLABORATIVE FILTERING ==========

def collaborative_recommendations(ratings_df, movies_df, user_id, n=5):
    """Simple user-based collaborative filtering"""
    # Convert user_id to int
    user_id = int(user_id)
    
    # Check if user exists
    if user_id not in ratings_df['userId'].unique():
        print(f"User {user_id} not found")
        return pd.DataFrame()
    
    # Get user's ratings
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    user_rated_movies = set(user_ratings['movieId'])
    
    # Find similar users
    similar_users = []
    for other_user in ratings_df['userId'].unique():
        if other_user == user_id:
            continue
        
        # Get overlap in ratings
        other_ratings = ratings_df[ratings_df['userId'] == other_user]
        common_movies = set(other_ratings['movieId']) & user_rated_movies
        
        if len(common_movies) < 5:  # Require at least 5 common movies
            continue
        
        # Calculate similarity (Pearson correlation)
        user_common = user_ratings[user_ratings['movieId'].isin(common_movies)]
        other_common = other_ratings[other_ratings['movieId'].isin(common_movies)]
        
        user_ratings_array = []
        other_ratings_array = []
        
        for movie_id in common_movies:
            user_rating = user_common[user_common['movieId'] == movie_id]['rating'].values[0]
            other_rating = other_common[other_common['movieId'] == movie_id]['rating'].values[0]
            
            user_ratings_array.append(user_rating)
            other_ratings_array.append(other_rating)
        
        # Calculate correlation
        similarity = np.corrcoef(user_ratings_array, other_ratings_array)[0, 1]
        if not np.isnan(similarity):
            similar_users.append((other_user, similarity))
    
    # Sort by similarity
    similar_users.sort(key=lambda x: x[1], reverse=True)
    top_similar_users = [user for user, _ in similar_users[:10]]  # Top 10 similar users
    
    # Get movies rated highly by similar users but not rated by the user
    candidate_movies = {}
    for similar_user, similarity in similar_users[:10]:
        similar_user_ratings = ratings_df[
            (ratings_df['userId'] == similar_user) & 
            (~ratings_df['movieId'].isin(user_rated_movies))
        ]
        
        # Only consider movies rated 4.0 or higher
        high_rated = similar_user_ratings[similar_user_ratings['rating'] >= 4.0]
        
        for _, row in high_rated.iterrows():
            movie_id = row['movieId']
            rating = row['rating']
            
            if movie_id not in candidate_movies:
                candidate_movies[movie_id] = {'score': 0, 'weighted_sum': 0}
            
            # Add weighted rating (rating * similarity)
            candidate_movies[movie_id]['weighted_sum'] += rating * similarity
            candidate_movies[movie_id]['score'] += similarity
    
    # Calculate predicted ratings
    predictions = []
    for movie_id, data in candidate_movies.items():
        if data['score'] > 0:
            predicted_rating = data['weighted_sum'] / data['score']
            predictions.append((movie_id, predicted_rating))
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = predictions[:n]
    
    # Get movie details
    result = []
    for movie_id, pred_rating in top_movies:
        movie = movies_df[movies_df['movieId'] == movie_id]
        if not movie.empty:
            title = movie['title'].values[0]
            genres = movie['genres'].values[0]
            result.append((title, genres, pred_rating))
    
    # Create DataFrame with results
    if result:
        df = pd.DataFrame(result, columns=['title', 'genres', 'predicted_rating'])
        return df
    else:
        return pd.DataFrame()

# ========== MAIN APPLICATION ==========

def main():
    """Main function to run the movie recommender"""
    print("Movie Recommender System")
    print("=======================\n")
    
    try:
        # Check if data exists
        if not os.path.exists("data/ml-latest-small/movies.csv"):
            movies_path, ratings_path = download_movielens_data()
        else:
            movies_path = "data/ml-latest-small/movies.csv"
            ratings_path = "data/ml-latest-small/ratings.csv"
        
        # Load the data
        print("Loading data...")
        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
        print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
        
        while True:
            print("\nRecommendation Options:")
            print("1. Get movie recommendations based on a movie you like")
            print("2. Get user-based recommendations")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                movie_title = input("Enter a movie title: ")
                print(f"\nFinding movies similar to '{movie_title}'...")
                recommendations = content_based_recommendations(movies_df, movie_title)
                
                if not recommendations.empty:
                    recommendations['similarity'] = recommendations['similarity'].round(3)
                    print("\nRecommended Movies:")
                    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                        print(f"{i}. {row['title']} ({row['genres']}) - Similarity: {row['similarity']}")
                else:
                    print("No recommendations found.")
            
            elif choice == '2':
                try:
                    user_id = int(input("Enter user ID: "))
                    print(f"\nFinding recommendations for User {user_id}...")
                    recommendations = collaborative_recommendations(ratings_df, movies_df, user_id)
                    
                    if not recommendations.empty:
                        recommendations['predicted_rating'] = recommendations['predicted_rating'].round(2)
                        print("\nRecommended Movies:")
                        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                            print(f"{i}. {row['title']} ({row['genres']}) - Predicted Rating: {row['predicted_rating']}")
                    else:
                        print("No recommendations found.")
                except ValueError:
                    print("Please enter a valid user ID.")
            
            elif choice == '3':
                print("Thank you for using the Movie Recommender System!")
                break
            
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
