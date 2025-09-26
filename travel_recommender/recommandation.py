import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import joblib

#Load model (vectorizer and sentiment models)
with open("models/tfidf_vectorizer.pkl","rb") as f:
    vectorizer = joblib.load(f)

with open("models/sentiment_model.pkl","rb") as f:
    sentiment_model = joblib.load(f)

# ------------------------------- Sentiment Analysis ----------------------------
def analyze_sentiment(comment):
    X = vectorizer.transform([comment])   
    prediction = sentiment_model.predict(X)[0]
    
    if prediction == -1:   # Negetive
        return -1
    elif prediction == 0: # neutral
        return 0
    else:                 # positive
        return 2


#------------------------------- Place Scores -----------------------------------
def get_place_scores(file_path):
    with open(file_path, "r") as f:
        reviews = json.load(f)
    
    place_scores = {}
    for place, entries in reviews.items():
        sentiments = []
        ratings = []
        
        for entry in entries:
            sentiments.append(analyze_sentiment(entry["comment"]))
            ratings.append(entry["rating"])
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        final_score = (0.5 * avg_sentiment) + (0.5 * (avg_rating / 5))
        
        place_scores[place] = {"avg_sentiment": avg_sentiment, "avg_rating": avg_rating, "final_score": final_score }
    return place_scores

def recommend_places(file_path, top_n=3):
    scores = get_place_scores(file_path)
    sorted_places = sorted(scores.items(), key=lambda x: x[1]["final_score"], reverse=True)
    return sorted_places[:top_n]

recommendations = recommend_places("data/reviews.json", top_n=3)
for place, score in recommendations:
    print(f"{place} → Score: {score['final_score']:.2f} (Rating: {score['avg_rating']}, Sentiment: {score['avg_sentiment']:.2f})")

# --------------------------- User Likes & Matrix ------------------------------
with open("data/user_likes.json") as f:
    likes_data = json.load(f)

rows = []
for place, data in likes_data.items():
    for user in data["users"]:
        rows.append({"user": user,"place":place,"like":1})

df_likes = pd.DataFrame(rows)

#Create user-item matrix
user_item_matrix = df_likes.pivot_table(index="user",columns="place",values="like", fill_value=0)


#-------------------------- Collaborative Filtering -----------------------------
user_similirity = cosine_similarity(user_item_matrix)
user_similirity_df = pd.DataFrame(user_similirity, index=user_item_matrix.index, columns=user_item_matrix.index )

item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity,index=user_item_matrix.columns, columns=user_item_matrix.columns)

svd = TruncatedSVD(n_components=3)
latent_matrix = svd.fit_transform(user_item_matrix)

# Similarity of users in latent space
latent_similarity = cosine_similarity(latent_matrix)
latent_similarity_df = pd.DataFrame(
    latent_similarity, 
    index=user_item_matrix.index, 
    columns=user_item_matrix.index)

# User - based CF
def ubcf_recommend(user,top_n=5):
    if user not in user_item_matrix.index:
        return []  #No recommondations for new user.

    similar_users = user_similirity_df[user].sort_values(ascending = False)[1:] # skip self
    recommendations = pd.Series(0, index=user_item_matrix.columns)
    for other_user, similarity_score in similar_users.items():
        recommendations += user_item_matrix.loc[other_user]*similarity_score
    recommendations = recommendations[user_item_matrix.loc[user] == 0]
    return recommendations.sort_values(ascending=False).head(top_n).index.tolist()

# Item - based CF
def ibcf_recommend(user, top_n=5):

    if user not in user_item_matrix.index:
        return []  #No recommondations for new user.
    
    liked_items = user_item_matrix.loc[user][user_item_matrix.loc[user]==1].index
    recommendations = pd.Series(0,index=user_item_matrix.columns)
    for item in liked_items:
        recommendations += item_similarity_df[item]
    recommendations = recommendations[user_item_matrix.loc[user] == 0]
    return recommendations.sort_values(ascending=False).head(top_n).index.tolist()

# SVD 
def svd_recommend(user, top_n=5):
    if user not in latent_similarity_df.index:
        print("user doesnt in matrix")
        return []  # User cold-start fallback elsewhere

    similar_users = latent_similarity_df[user].sort_values(ascending=False)[1:]
    recommendations = pd.Series(0, index=user_item_matrix.columns)
    for other_user, similarity_score in similar_users.items():
        recommendations += user_item_matrix.loc[other_user] * similarity_score
    recommendations = recommendations[user_item_matrix.loc[user] == 0]
    return recommendations.sort_values(ascending=False).head(top_n).index.tolist()

# ------------------------------- Hybrid Recommendation -------------------------
def hybrid_recommend(user, reviews_file, top_n=10, alpha=0.4, beta=0.3, gamma=0.3):
    sentiment_scores = get_place_scores(reviews_file)

    ubcf_places = ubcf_recommend(user, top_n=top_n*3)
    ibcf_places = ibcf_recommend(user, top_n=top_n*3)
    svd_places  = svd_recommend(user, top_n=top_n*3)

    # Cold-start fallback: if all CF methods return empty
    if not ubcf_places and not ibcf_places and not svd_places:
        sorted_places = sorted(sentiment_scores.items(), key=lambda x: x[1]["final_score"], reverse=True)
        return sorted_places[:top_n]

    all_places = list(dict.fromkeys(ubcf_places + ibcf_places + svd_places))

    hybrid_scores = {}
    for place in all_places:
        ubcf_score = (len(ubcf_places) - ubcf_places.index(place)) / len(ubcf_places) if place in ubcf_places else 0
        ibcf_score = (len(ibcf_places) - ibcf_places.index(place)) / len(ibcf_places) if place in ibcf_places else 0
        svd_score  = (len(svd_places) - svd_places.index(place)) / len(svd_places) if place in svd_places else 0

        cf_score = alpha * ubcf_score + beta * ibcf_score + gamma * svd_score
        sr_score = sentiment_scores.get(place, {}).get("final_score", 0)

        hybrid_scores[place] = 0.7 * cf_score + 0.3 * sr_score

    sorted_places = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = [place for place, score in sorted_places[:top_n]]

    # Fill missing recommendations if needed
    if len(recommendations) < top_n:
        sorted_sentiment = sorted(sentiment_scores.items(), key=lambda x: x[1]["final_score"], reverse=True)
        for place, score in sorted_sentiment:
            if place not in recommendations:
                recommendations.append(place)
            if len(recommendations) == top_n:
                break

    return recommendations

# ----------------------------------- Cold Start --------------------------------
def get_popular_places(top_n=5):
    place_counts = df_likes.groupby("place")["like"].sum().sort_values(ascending=False)
    return list(place_counts.head(top_n).index)

def get_recommendations_for_user(user):
    #handling COLD-START for new users
    if user not in user_item_matrix.index:
        return get_popular_places(top_n=10)
    return hybrid_recommend(user, "data/reviews.json", top_n=10)
    

# -------------------- Example --------------------
if __name__ == "__main__":
    print(get_recommendations_for_user("k"))