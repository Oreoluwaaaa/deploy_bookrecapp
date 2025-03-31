import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer as sentran

#isbn cleaning
def isbn10_to_isbn13(isbn10):
    if len(isbn10) < 10:
        isbn10 = isbn10.zfill(10)  # pad with leading zeros if necessary
    
    # add prefix '978' to convert ISBN-10 to ISBN-13
    isbn13_prefix = '978' + isbn10[:-1]
    
    # calculate the check digit for ISBN-13
    check_sum = sum((1 if i % 2 == 0 else 3) * int(digit) for i, digit in enumerate(isbn13_prefix))
    check_digit = (10 - (check_sum % 10)) % 10
    
    return isbn13_prefix + str(check_digit)

#function to clean up genre information in the data frame
def genre_str_to_arr(genre):
    genre = genre[2:-2].replace("\'", "")
    genre = genre.split(", ")

    return genre

#function to find cosine similarity
def sim_score_(emb1, emb2):
    return np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))

#get books data
def getting_books_data(books):
    #fill empty isbn values with integer representation
    books_dataframe = pd.read_csv(books)
    books_dataframe['isbn'] = books_dataframe['isbn'].fillna('')

    #booksdf creation
    booksdf = books_dataframe.copy()
    booksdf = booksdf[["title", "authors", "average_rating", "book_id","original_publication_year", "genres", "pages", "description", "image_url"]]

    #change year column to basic integer
    booksdf.loc[:, 'original_publication_year'] = booksdf['original_publication_year'].fillna(-1)
    booksdf.loc[:, 'original_publication_year'] = booksdf['original_publication_year'].astype(int)

    #change isbn10 to 13 with the function above

    booksdf.loc[:, 'isbn13'] = books_dataframe['isbn'].apply(isbn10_to_isbn13)

    #add isbn13 to the dataframe
    booksdf = booksdf[["title", "authors", "book_id","isbn13", "average_rating","original_publication_year", "genres", "pages", "description", "image_url"]]

    #change genre to array in dataframe
    booksdf.loc[:, "genres"] = booksdf["genres"].apply(genre_str_to_arr)
    
    #change title to remove anything with and within parentheses
    booksdf.loc[:, "title"] = booksdf.loc[:, "title"].str.replace(r'\s*\(.*?\)\s*', '', regex=True)

    return booksdf

#load the model
def load_model():
    return sentran('sentence-transformers/all-MiniLM-L6-v2')

#generate embeddings using descriptions
def generate_desc_embeddings(books_dataframe, sbert_model):
    descriptions = books_dataframe["description"].tolist()

    # define batch size
    batch_size = 128

    # compute embeddings in batches
    embeddings = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i+batch_size]
        batch_embeddings = sbert_model.encode(batch)
        embeddings.append(batch_embeddings)

    # concatenate all batches into a single array
    embeddings = np.vstack(embeddings)

    return embeddings


#generate embeddings based on genre
def generate_genre_embeddings(books_dataframe, genre_map):
    genre_emb = {}
    
    for i, genres in enumerate(books_dataframe["genres"]):
        genre_emb[i] = [float(0)] * 39


    #math for genre ratios
    for i, genres in enumerate(books_dataframe["genres"]):
        l = len(genres)
        for k, genre in enumerate(genres): 
            genre_emb[i][genre_map[genre]] = (l-k)*100/sum(range(l + 1))
    
    #final 2d genre list
    genre_list = [0] * len(genre_emb)
    for key, values in genre_emb.items():
        genre_list[key]= values

    return genre_list

#generate clusters based on genre embeddings
def generate_clusters(genre_list):
    optimal_k = 12
    kmeans_genre = KMeans(n_clusters = optimal_k, random_state = 2, n_init="auto")
    kmeans_genre.fit(genre_list)

    #getting clusters
    cluster_map = {}
    for idx, clusteridx in enumerate(kmeans_genre.labels_):
        if clusteridx in cluster_map:
            cluster_map[clusteridx].append(idx)
        else:
            cluster_map[clusteridx] = [idx]

    return cluster_map, kmeans_genre

#predict a new book given an index  
def predict_new_book(new_bk_idx, booksdf, genre_map, kmeans_genre, cluster_map, sbert_model, embeddings):
    newdesc = booksdf.iloc[new_bk_idx]["description"]
    newtitle = booksdf.iloc[new_bk_idx]["title"]
    newgenre = booksdf.iloc[new_bk_idx]["genres"]
    newimage = booksdf.iloc[new_bk_idx]["image_url"]
    newgenre_emb = [float(0)] * 39

    #get the new genre embedding
    for k, genre in enumerate(newgenre): 
        l = len(newgenre)
        newgenre_emb[genre_map[genre]] = (l-k)*100/sum(range(l + 1))


    #find its cluster
    cluster = kmeans_genre.predict([newgenre_emb])[0]
   
    #new entry embedding based on description
    newbook_emb = sbert_model.encode(newdesc)


    # find cosine similarity
    similarity_score = [sim_score_(emb, newbook_emb) for emb in embeddings]

    #sort keeping index
    recsidx = np.array(similarity_score).argsort()[::-1][:100]

    #filtering genre cluster info through book description cosine similary info
    i, recommendations = 0, []
    for idx in recsidx:
        if idx in cluster_map[cluster] and i < 10:
            i += 1
            title = booksdf.loc[idx]["title"]
            isbn13 = booksdf.loc[idx]["isbn13"]
            rating = booksdf.loc[idx]["average_rating"]
            image = booksdf.iloc[idx]["image_url"]
            recommendations.append((title, isbn13, rating, image))
            

    return sort_recs(recommendations, booksdf) 

#get user genre preferences
def get_user_genre(genre_map, userGenreInput):
    user_genre = [float(0)] * 39
    user_genre_input, i = [], 0
    
    for input in userGenreInput:
        if input != "":
            user_genre_input.append(input)

    num = len(user_genre_input)

    for k, genre in enumerate(user_genre_input): 
        user_genre[genre_map[genre]] = (num-k)*100/sum(range(num + 1))

    return user_genre

#predict book based on given user genre
def predict_book_with_genre(user_genre, kmeans_genre, cluster_map, booksdf):
    #getting what cluster it belongs to
    cluster = kmeans_genre.predict([user_genre])[0]
    recommendations = []

    for i in cluster_map[cluster]:
        title = booksdf.loc[i]["title"]
        isbn13 = booksdf.loc[i]["isbn13"]
        rating = booksdf.loc[i]["average_rating"]
        image = booksdf.iloc[i]["image_url"]
        recommendations.append((title, isbn13, rating, image, i))
    
    #display top 10 recommendations
    top_recs = recommendations[:10]
    print("Here are 10 recommendations based on your genre preferences:")
    for idx, (title, isbn13, rating, image, i) in enumerate(top_recs):
        print(f"Index: {i}. {title} (ISBN13: {isbn13}, Rating: {rating})")

    return top_recs


#predict book based on user description
def predict_book_with_description(userDesc, booksdf, sbert_model, embeddings):
    #new entry embedding based on description
    newbook_emb = sbert_model.encode(userDesc)


    # find cosine similarity
    similarity_score = [sim_score_(emb, newbook_emb) for emb in embeddings]

    #sort keeping index
    recsidx = np.array(similarity_score).argsort()[::-1][:100]

    #filtering genre cluster info through book description cosine similary info
    recommendations = []
    for idx in recsidx[:10]:
        title = booksdf.loc[idx]["title"]
        isbn13 = booksdf.loc[idx]["isbn13"]
        rating = booksdf.loc[idx]["average_rating"]
        image = booksdf.iloc[idx]["image_url"]
        recommendations.append([title, isbn13, rating, image, idx])
        
    top_recs = recommendations
    print("Here are 10 recommendations based on your genre preferences:")
    for idx, (title, isbn13, rating, image, i) in enumerate(top_recs):
        print(f"Index: {i}. {title} (ISBN13: {isbn13}, Rating: {rating})")

    return top_recs

#final rec method
def finalrec(top_recs):
    chosen_indices = []
    while len(chosen_indices) < 3:
        new_bk_idx = int(input(f"Choose the index of the book you liked (available: {', '.join(str(x[3]) for x in top_recs)}): "))
        if new_bk_idx in [x[3] for x in top_recs]:  # check if the index is valid
            chosen_indices.append(new_bk_idx)
        else:
            print("Invalid index, please try again.")

    return chosen_indices

# sort the recommendations by their average rating
def sort_recs(recommendations, booksdf):
    sorted_recs = sorted(recommendations, key=lambda x: x[2], reverse=True)  # x[2] is the rating
    
    # Display the sorted recommendations
    print("Final Recommendations sorted by rating:")
    for title, isbn13, rating, image in sorted_recs:
        print(f"{title} (ISBN13: {isbn13}, Rating: {rating})")
    
    return sorted_recs