try:
    import sys # Just in case
    start = sys.version.index('|') # Do we have a modified sys.version?
    end = sys.version.index('|', start + 1)
    version_bak = sys.version # Backup modified sys.version
    sys.version = sys.version.replace(sys.version[start:end+1], '') # Make it legible for platform module
    import platform
    platform.python_implementation() # Ignore result, we just need cache populated
    platform._sys_version_cache[version_bak] = platform._sys_version_cache[sys.version] # Duplicate cache
    sys.version = version_bak # Restore modified version string
except ValueError: # Catch .index() method not finding a pipe
    pass
import pandas as pd

metadata = pd.read_csv('./the-movies-dataset/movies_metadata.csv', low_memory=False)

# Print the first three rows
# print(metadata.head(3))

C= metadata['vote_average'].mean()
print(C)

m=metadata['vote_count'].quantile(0.90)
print(m)

filter_movies=metadata.copy().loc[metadata['vote_count']>=m]

print(filter_movies.shape)


def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
filter_movies['score'] = filter_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
filter_movies = filter_movies.sort_values('score', ascending=False)

#Print the top 15 movies
print(filter_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))