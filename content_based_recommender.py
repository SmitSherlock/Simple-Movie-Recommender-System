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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


metadata = pd.read_csv('./the-movies-dataset/movies_metadata.csv', low_memory=False)

# print(metadata['overview'].head())
#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#Output the shape of tfidf_matrix
# print(tfidf_matrix.shape)


# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


#Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]

    
get_recommendations('The Dark Knight Rises')