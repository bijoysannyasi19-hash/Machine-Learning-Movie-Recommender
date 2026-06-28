import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    """
    A content-based movie recommender using TF-IDF on Overview and Multi-hot encoding on Genre.
    """
    def __init__(self, data_path='mymoviedb.csv'):
        self.data_path = data_path
        self.df_movies = None
        self.similarity_matrix = None
        self._prepare_data()

    def _prepare_data(self):
        """Loads data, deduplicates it to movie level, and computes the similarity matrix."""
        df_raw = pd.read_csv(self.data_path, lineterminator='\n')
        
        # Deduplicate to movie-level based on Title
        # For simplicity, we just take the first occurrence if titles collide
        self.df_movies = df_raw.groupby('Title').agg({
            'Overview': 'first',
            'Genre': lambda x: ', '.join(list(set(x.iloc[0].split(', '))))
        }).reset_index()

        # Fill NaNs if any
        self.df_movies['Overview'] = self.df_movies['Overview'].fillna('')
        self.df_movies['Genre'] = self.df_movies['Genre'].fillna('')

        # 1. Vectorize Overview (Text)
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        overview_matrix = tfidf.fit_transform(self.df_movies['Overview'])

        # 2. Vectorize Genre (Multi-hot)
        # Using TfidfVectorizer with custom tokenizer to handle comma separated genres
        # acts identically to multi-hot if we don't use idf or if we use it, it gives weight to rare genres
        genre_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(', '), token_pattern=None)
        genre_matrix = genre_vectorizer.fit_transform(self.df_movies['Genre'])

        # Combine both features
        # We can stack them horizontally, weighting genre a bit higher. Let's give genre a 2x weight
        import scipy.sparse as sp
        combined_matrix = sp.hstack([overview_matrix, genre_matrix * 2.0])

        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(combined_matrix)

    def recommend(self, title: str, n: int = 5) -> list[str]:
        """
        Returns the top-N most similar movie titles, excluding the input movie itself.
        If exact title isn't found, uses fuzzy matching to suggest the closest valid title.
        """
        # Check if title exists
        if title not in self.df_movies['Title'].values:
            closest_matches = difflib.get_close_matches(title, self.df_movies['Title'].values, n=1)
            if closest_matches:
                closest_title = closest_matches[0]
                print(f"Title '{title}' not found. Did you mean '{closest_title}'?")
                title = closest_title
            else:
                return [f"Error: Could not find '{title}' or any close matches."]
        
        # Get index of the movie
        idx = self.df_movies.index[self.df_movies['Title'] == title][0]

        # Get similarity scores for this movie
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort the movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the n most similar movies (ignoring the first one which is itself)
        sim_scores = sim_scores[1:n+1]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top n most similar movies
        return self.df_movies['Title'].iloc[movie_indices].tolist()

if __name__ == '__main__':
    # Simple test
    recommender = MovieRecommender()
    print("Recommendations for 'Spider-Man: No Way Home':")
    print(recommender.recommend("Spider-Man: No Way Home"))
