# Netflix Movie Data Analysis & Recommendation System

This project explores a dataset of ~9,800 movies, extracting insights on popularity, genres, and providing interactive and machine-learning tools based on the dataset. It has been unified into a single, cohesive workflow.

## The Unified Notebook

- **`main_analysis.py`**: This master script combines all data processing, EDA, Machine Learning, and Computer Vision tasks into one seamless flow:
  - **1 & 2. Data Cleaning & Basic EDA**: Cleans `mymoviedb.csv` and answers fundamental exploratory questions.
  - **4. Advanced Correlation Analysis & Genre Trends**: Performs correlation analysis on popularity and votes, normalizes heavily right-skewed distributions, and charts the top 5 genres over time.
  - **5. Machine Learning Classifier for Popularity**: Prepares multi-hot encoded genres and trains a Random Forest Classifier to predict the `Vote_Average` quartile category, saving the best model into `models/`.
  - **6. Content-Based Movie Recommender**: Showcases how to fetch similar movies using TF-IDF and Cosine Similarity, including fuzzy string matching for typos.
  - **7. Computer Vision: Poster Color Analysis**: Dynamically downloads movie posters via `requests` and applies K-Means clustering to determine and analyze the dominant color/brightness for top genres.
  
  *Run command*: `python main_analysis.py` (Charts will automatically save to the `visualizations/` folder)

## The Interactive Dashboard

- **`app.py`**: A robust Streamlit dashboard allowing users to filter movies by genre, release year, and vote count, featuring interactive KPI cards and Plotly charts.
  - *Run command*: `streamlit run app.py`
  - *(Include a screenshot of the dashboard running here for your portfolio!)*

## Testing & Automation

- **`tests/` & GitHub Actions CI**: Contains `pytest` unit tests validating the categorization logic and recommender behavior. The CI pipeline ensures code stability automatically upon push.
  - *Run command*: `pytest tests/`

## Setup

1. Clone the repository.
2. Ensure you have Python 3.10+ installed.
3. Install dependencies: `pip install -r requirements.txt`

## Project Report

A highly detailed summary of the implementations and findings is provided in `project_report.pdf`.
