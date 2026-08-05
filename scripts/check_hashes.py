import pandas as pd

p = "data/Bangalore Weather Data (Visual Crossing Weather).csv"
try:
    df = pd.read_csv(p, dtype=str)
except Exception as e:
    print("ERROR_READING_CSV", e)
    raise

# Find rows where any column is composed only of one or more '#' characters
mask = df.apply(lambda col: col.str.match(r'^#+$'), axis=0).any(axis=1)
result = df[mask]

if result.empty:
    print("NO_HASH_ROWS")
else:
    print("HASH_ROWS_COUNT:", len(result))
    print("HASH_ROW_INDICES:", result.index.tolist())
    # print a small sample of the problematic rows
    print(result.head(20).to_csv(index=True))
