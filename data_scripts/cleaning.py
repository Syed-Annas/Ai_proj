df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.drop(columns=['id'], inplace=True)