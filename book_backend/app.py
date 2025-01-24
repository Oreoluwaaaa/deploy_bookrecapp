import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  
from recSystem import ( 
    getting_books_data,
    load_model,
    generate_genre_embeddings,
    generate_clusters,
    get_user_genre,
    predict_book_with_description,
    predict_book_with_genre,
    predict_new_book
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# load your data, model and saved embeddings
booksdf = getting_books_data("./books.csv")
sbert_model = load_model()
desc_embeddings = np.load("./desc_embeddings.npy")

#initalise genre map
genre_map = {
    'art': 0, 'biography': 1, 'books': 2, 'business': 3,
    'chick-lit': 4, 'christian': 5, 'classics': 6, 'comics': 7,
    'contemporary': 8, 'cookbooks': 9, 'crime': 10, 'fantasy': 11,
    'fiction': 12, 'gay-and-lesbian': 13, 'graphic-novels': 14,
    'historical-fiction': 15, 'history': 16, 'horror': 17,
    'humor-and-comedy': 18, 'manga': 19, 'memoir': 20, 'music': 21,
    'mystery': 22, 'nonfiction': 23, 'paranormal': 24, 'philosophy': 25,
    'poetry': 26, 'psychology': 27, 'religion': 28, 'romance': 29,
    'science': 30, 'science-fiction': 31, 'self-help': 32,
    'spirituality': 33, 'sports': 34, 'suspense': 35, 'thriller': 36,
    'travel': 37, 'young-adult': 38
}

#initialise genrelist
genre_list = generate_genre_embeddings(booksdf, genre_map)

#get cluster map and kmeans
cluster_map, kmeans_genre = generate_clusters(genre_list)


#app route for description recommendations
@app.route('/recommendations/description', methods=['POST'])
def descRecommendation():
    data = request.get_json()
    userDesc = data.get('description')

    recommendations = predict_book_with_description(userDesc, booksdf, sbert_model, desc_embeddings)

    recommendations_serializable = [
        {
            "title": title,
            "isbn13": str(isbn13),  
            "rating": float(rating),  
            "image": str(image),
            "index": int(i)  
        }
        for (title, isbn13, rating, image, i) in recommendations
    ]
    return jsonify(recommendations_serializable)


#app route for genre recommendations
@app.route('/recommendations/genre', methods=['POST'])
def genreRecommendation():
    data = request.get_json()
    userGenre = data.get('genres')

    user_genre=get_user_genre(genre_map, userGenre)
    recommendations = predict_book_with_genre(user_genre, kmeans_genre, cluster_map, booksdf)

    recommendations_serializable = [
        {
            "title": title,
            "isbn13": str(isbn13),  
            "rating": float(rating), 
            "image": str(image),
            "index": int(i)  
        }
        for (title, isbn13, rating, image, i) in recommendations
    ]

    return jsonify(recommendations_serializable)

#app route for final recommendations
@app.route('/recommendations/final-recs', methods=['POST'])
def finalRecommendation():
    data = request.get_json()
    chosenIndices = data.get('chosenIndices')
    recommendations = []
    for index in chosenIndices:
        recommendations.extend(predict_new_book(index, booksdf, genre_map, kmeans_genre, cluster_map, sbert_model, desc_embeddings))

    print(recommendations[0])
    recommendations_serializable = [
        {
            "title": title,
            "isbn13": str(isbn13), 
            "rating": float(rating), 
            "image": str(image),
        }
        for (title, isbn13, rating, image) in recommendations
    ]

    return jsonify(recommendations_serializable)

if __name__ == '__main__':
    app.run(debug=True)
