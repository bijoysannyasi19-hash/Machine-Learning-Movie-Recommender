import pytest
from recommender import MovieRecommender

@pytest.fixture(scope="module")
def recommender():
    # Uses the local mymoviedb.csv for tests. 
    # In a real environment, you might use a mocked smaller csv.
    return MovieRecommender(data_path='mymoviedb.csv')

def test_recommend_returns_n_items(recommender):
    res = recommender.recommend("Spider-Man: No Way Home", n=3)
    assert len(res) == 3

def test_recommend_excludes_input_title(recommender):
    title = "The Batman"
    res = recommender.recommend(title, n=10)
    assert title not in res

def test_recommend_fuzzy_matching(recommender):
    # Intentional typo
    typo_title = "The Batmen"
    res = recommender.recommend(typo_title, n=5)
    
    # Should not return an error string
    assert len(res) == 5
    assert "Error:" not in res[0]

def test_recommend_completely_unknown_title(recommender):
    # A title that definitely won't match anything
    res = recommender.recommend("Aalskdjfhgqpqpqpzxcvbnm", n=5)
    assert len(res) == 1
    assert "Error: Could not find" in res[0]
