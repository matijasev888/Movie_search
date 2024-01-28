from flask import Flask, render_template, request, jsonify
import pandas as pd
from movie_filter import filter, predict_genre, get_model
from WordMatcher import WordMatcher

app = Flask(__name__)

file_path = 'movies.csv'
movies_df = pd.read_csv(file_path)

# Initialize WordMatcher with your words.csv
word_matcher = WordMatcher('words.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    query = request.args.get('search', '')
    genre_prediction = None

    classifier = get_model()

    if query:
        filtered_movies = filter(query)
        genre_prediction = predict_genre(classifier, query)
    else:
        filtered_movies = movies_df

    start = (page - 1) * per_page
    end = start + per_page
    total = len(filtered_movies)
    pages = total // per_page
    if total % per_page != 0:
        pages += 1

    # Calculate end_page for pagination
    start_page = page
    end_page = min(pages, start_page + 9)

    movies_to_display = filtered_movies.iloc[start:end]

    return render_template('index.html', movies=movies_to_display, query=query, page=page, pages=pages, start_page=start_page, end_page=end_page, genre_prediction=genre_prediction)

@app.route('/suggest', methods=['GET'])
def suggest():
    query = request.args.get('query', '')
    suggestions = word_matcher.find_close_matches(query, num_matches=5)
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True)
