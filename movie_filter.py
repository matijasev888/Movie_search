import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
import spacy
import re
import jellyfish
import math
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy's pre-trained English model
nlp = spacy.load('en_core_web_sm')

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

movie_data = pd.read_csv('movies.csv')
encoded_movies = np.load('encoded_movies.npy')

def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length').to('cpu')
    outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].detach().numpy()

def spacy_lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def extract_human_names(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

def extract_years(text):
    return re.findall(r'\b\d{4}\b', text)

def name_similarity(name1, name2):
    return jellyfish.jaro_winkler_similarity(name1, name2)

def year_similarity(year1, year2, decay_rate=0.07):
    year1_int = int(year1)
    year2_int = int(year2)
    year_difference = abs(year1_int - year2_int)
    similarity = np.exp(-decay_rate * year_difference)
    return similarity

def calculate_movie_score(extracted_names, extracted_year, movie_names, movie_year, cosine_similarity, name_threshold=0.8, year_threshold=0.8):
    # Calculate name similarities
    name_similarities = [name_similarity(extracted_name, movie_name) for extracted_name in extracted_names for movie_name in movie_names if name_similarity(extracted_name, movie_name) >= name_threshold]
    name_score_sum = sum(name_similarities)
    name_score_count = len(name_similarities)
    # Calculate year similarity
    year_score = year_similarity(extracted_year, movie_year) if year_similarity(extracted_year, movie_year) >= year_threshold else 0

    # Calculate the final score
    score = math.log(name_score_count * name_score_sum * 1.33  + math.e) * (1 + year_score**2 * math.e) * (1 + cosine_similarity**2*5)
    return score

# Function to find similar movies
def filter(description, encoded_movies=encoded_movies, top_n=20):
    preprocessed_description = spacy_lemmatize_text(description.lower())
    extracted_names = extract_human_names(preprocessed_description)
    extracted_years = extract_years(preprocessed_description)

    encoded_description = encode_text(preprocessed_description)  # Assuming encode_text function is defined
    cosine_similarities = cosine_similarity(encoded_description, encoded_movies).flatten()

    new_scores = []
    for index, row in movie_data.iterrows():
        movie_names = row['names'].split(',')  # Assuming names are comma-separated
        movie_year = str(row['year'])

        # Use the first extracted year for simplicity
        extracted_year = extracted_years[0] if extracted_years else 0

        score = calculate_movie_score(extracted_names, extracted_year, movie_names, movie_year, cosine_similarities[index])
        new_scores.append(score)

    top_indices = np.argsort(new_scores)[-top_n:][::-1]
    similar_movies = movie_data.iloc[top_indices].copy()
    similar_movies['score'] = np.array(new_scores)[top_indices]

    return similar_movies
