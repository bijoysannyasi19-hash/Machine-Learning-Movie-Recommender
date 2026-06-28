# %% [markdown]
# # ML Classifier for Popularity Category
# This notebook builds a supervised machine learning model to predict the `Vote_Average` quartile category (`not_popular`, `below_avg`, `average`, `popular`) using features like `Vote_Count`, `Release_Year`, and `Genre`.

# %%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# %% [markdown]
# ## 1. Data Preparation
# We'll load the data and recompute the target category based on quartiles. 

# %%
def categorize_column(df, col, labels):
    """Bucket a numeric column into quartile-based categories."""
    edges = [df[col].min() - 0.001, df[col].quantile(.25), df[col].quantile(.5),
             df[col].quantile(.75), df[col].max() + 0.001]
    return pd.cut(df[col], bins=edges, labels=labels, duplicates='drop')

def main():
    print("Loading data...")
    df_raw = pd.read_csv('mymoviedb.csv', lineterminator='\n')

    df_raw['Release_Year'] = pd.to_datetime(df_raw['Release_Date']).dt.year

    # Deduplicate to movie-level
    df_movies = df_raw.groupby(['Title', 'Release_Year']).agg({
        'Vote_Count': 'first',
        'Vote_Average': 'first',
        'Genre': lambda x: list(set(x.iloc[0].split(', ')))
    }).reset_index()

    labels = ['not_popular', 'below_avg', 'average', 'popular']
    df_movies['Vote_Average_Category'] = categorize_column(
        df_movies, 'Vote_Average', labels
    )

    df_movies.dropna(subset=['Vote_Average_Category'], inplace=True)
    
    # %% [markdown]
    # ## 2. Feature Engineering
    # We will multi-hot encode the `Genre` column.
    
    # %%
    mlb = MultiLabelBinarizer()
    genre_features = pd.DataFrame(mlb.fit_transform(df_movies['Genre']), columns=mlb.classes_, index=df_movies.index)

    X = pd.concat([df_movies[['Vote_Count', 'Release_Year']], genre_features], axis=1)
    y = df_movies['Vote_Average_Category']

    # 80/20 train/test split with fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # %% [markdown]
    # ## 3. Model Training & Evaluation
    # We'll compare Logistic Regression and Random Forest.
    
    # %%
    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # Evaluation Function
    def evaluate(model_name, y_true, y_pred):
        print(f"--- {model_name} ---")
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.4f}")
        report = classification_report(y_true, y_pred, output_dict=True)
        print(classification_report(y_true, y_pred))
        return report['macro avg']['f1-score']

    lr_f1 = evaluate("Logistic Regression", y_test, lr_preds)
    rf_f1 = evaluate("Random Forest Classifier", y_test, rf_preds)

    # %% [markdown]
    # ## 4. Conclusion
    # Random Forest typically performs better because it captures non-linear relationships and complex interactions between genres and vote counts much better than a simple linear model. Let's visualize the confusion matrix for the winning model.
    
    # %%
    if rf_f1 > lr_f1:
        winner_name = "Random Forest"
        winner_model = rf_model
        winner_preds = rf_preds
    else:
        winner_name = "Logistic Regression"
        winner_model = lr_model
        winner_preds = lr_preds

    print(f"Winner: {winner_name}")

    cm = confusion_matrix(y_test, winner_preds, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix: {winner_name}')
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    # Use plt.savefig instead of plt.show() to prevent blocking
    plt.savefig('models/confusion_matrix.png')
    print("Saved confusion matrix to models/confusion_matrix.png")

    # %% [markdown]
    # ## 5. Save the Model
    # We use joblib to save the best model to the `models/` directory for potential future use.
    
    # %%
    os.makedirs('models', exist_ok=True)
    joblib.dump(winner_model, 'models/vote_category_classifier.pkl')
    print("Model saved to models/vote_category_classifier.pkl")

if __name__ == '__main__':
    main()
