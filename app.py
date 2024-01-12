from flask import Flask, render_template, request
import pandas as pd
import json
from movie_filter import filter  # Import the filter function from filter.py

app = Flask(__name__)

file_path = 'movies.csv'
movies_df = pd.read_csv(file_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    page = request.args.get('page', 1, type=int)
    per_page = 20  # Adjust as needed
    query = request.args.get('search', '')

    if query:
        filtered_movies = filter(query)
    else:
        filtered_movies = movies_df

    # Implement pagination
    start = (page - 1) * per_page
    end = start + per_page
    total = len(filtered_movies)
    pages = total // per_page
    if total % per_page != 0:
        pages += 1

    movies_to_display = filtered_movies.iloc[start:end]

    return render_template('index.html', movies=movies_to_display, query=query, page=page, pages=pages)

if __name__ == '__main__':
    app.run(debug=True)
