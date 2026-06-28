import pytest
import pandas as pd
# Import the function from modeling (or we can just test the categorization logic generically)
from modeling import categorize_column

def test_categorize_column():
    df = pd.DataFrame({'Vote_Average': [2.0, 5.0, 7.0, 9.0]})
    labels = ['not_popular', 'below_avg', 'average', 'popular']
    
    # Bucket based on the values
    res = categorize_column(df, 'Vote_Average', labels)
    
    # Min is ~2.0, Max is ~9.0, Quartiles at ~3.5, 6.0, 8.0
    assert len(res) == 4
    # All rows should be assigned a category
    assert not res.isna().any()
    # The lowest should be not_popular
    assert res.iloc[0] == 'not_popular'
    # The highest should be popular
    assert res.iloc[3] == 'popular'
