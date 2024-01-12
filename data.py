import pandas as pd
import imdb
from tqdm import tqdm

# Load the dataset
file_path = 'popular_movies_subset.csv'  # Replace with your file path
movies_df = pd.read_csv(file_path)

# Create an instance of the IMDb class
ia = imdb.IMDb()

# Function to get movie poster URL from IMDb ID
def get_movie_poster_url(imdb_id):
    try:
        # Remove 'tt' prefix and fetch the movie
        movie = ia.get_movie(imdb_id[2:])

        # Return the full-size poster URL
        return movie.get('full-size cover url')
    except Exception as e:
        return None

# Applying the function to each row in the DataFrame and tracking progress
tqdm.pandas(desc="Fetching Poster URLs")
movies_df['poster_url'] = movies_df['imdb_id'].progress_apply(get_movie_poster_url)

# Save the updated DataFrame to a new CSV file
updated_file_path = 'updated_movies_metadata.csv'  # Replace with your desired file path
movies_df.to_csv(updated_file_path, index=False)

print("Updated dataset saved to:", updated_file_path)
