from flask import Flask, render_template,request, redirect,url_for,session
import csv
import os
import json
from urllib.parse import unquote #to handle %20 in urls
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommandation import get_recommendations_for_user

app = Flask(__name__)
app.secret_key = 'your secret key' #needed for session

def load_destinations():
    locations = []
    file_path = os.path.join("data", "locations.csv")
    with open(file_path, newline='', encoding='latin-1') as file:
        reader = csv.DictReader(file)
        for row in reader:
            locations.append(row)
    return locations

def load_users():
    file_path = os.path.join("data", "users.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return {}

def save_user(username,password):
    file_path = os.path.join("data", "users.json")
    users = load_users()
    users[username] = password
    with open(file_path,"w") as file:
        json.dump(users,file,indent=4)

def load_reviews():
    file_path = os.path.join("data", "reviews.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return {}

def save_reviews(reviews):
    file_path = os.path.join("data", "reviews.json")
    with open(file_path, "w") as file:
        json.dump(reviews, file, indent=4)

def load_likes():
    file_path = os.path.join("data", "user_likes.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return {}

def save_likes(likes):
    file_path = os.path.join("data", "user_likes.json")
    with open(file_path, "w") as file:
        json.dump(likes, file, indent=4)

#content-based Recommandation
def get_similar_destinations(selected_place,all_locations,top_n=5):
    combined_fetures = [
        f"{loc['type']} {loc['activity']} {loc['climate']} {loc['budget']}".lower()
        for loc in all_locations
    ]
    #Vectorize the text using TF-IDE
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_fetures)

    #Get the index of the selected place
    selected_place = next((i for i, loc in enumerate(all_locations)
                           if loc['place'].strip().lower()== selected_place.strip().lower()),None)
    if selected_place is None:
        return []
    #compute cosine similarity
    similar_score = cosine_similarity(tfidf_matrix[selected_place],tfidf_matrix).flatten()

    #sort by similarity
    similar_indices = similar_score.argsort()[::-1][1:top_n+1]
    #return top N similar destinations
    return [all_locations[i] for i in similar_indices]

#Home route
@app.route("/")
def index():
    username = session.get("username","Guest")

# If "Show Recommendations" was clicked
    if request.args.get("recommend") == "true":
        if username == "Guest":
            return redirect(url_for("login"))
        try:
            recommended_places = get_recommendations_for_user(username)
            if not recommended_places:
                no_recommendations = True
                locations = []
            else:
                print("Recommended places:", recommended_places)
                locations = [
                    loc for loc in load_destinations()
                    if loc["place"] in recommended_places
                ]
                show_recommendations =True
        except Exception as e:
            print("Error getting recommendations:", e)
            locations = load_destinations()
    else:
        locations = load_destinations()

    # Check if recommendation was requested before login
    if session.get("want_recommend") and username != "Guest":
        try:
            recommended_places = get_recommendations_for_user(username)
            recommended_locations = [
                loc for loc in load_destinations()
                if loc["place"] in recommended_places
            ]
            locations += [loc for loc in recommended_locations if loc not in locations]
            session.pop("want_recommend")  # clear flag
        except Exception as e:
            print("Error getting recommendations:", e)

    # Get filter values from request
    selected_type = request.args.get("type", "").strip()
    selected_climate = request.args.get("climate", "").strip()
    selected_budget = request.args.get("budget", "").strip()

    # Apply filters
    if selected_type:
        locations = [loc for loc in locations if selected_type.lower() in loc["type"].lower()]
    if selected_climate:
        locations = [loc for loc in locations if selected_climate.lower() in loc["climate"].lower()]
    if selected_budget:
        locations = [loc for loc in locations if selected_budget.lower() in loc["budget"].lower()]

    #pagination
    page = int(request.args.get("page", 1))
    per_page = 12  # 5 images × 2 rows
    start = (page-1)* per_page
    end = start +per_page
    total_pages = (len(locations) + per_page - 1) // per_page  # round up
    paginated = locations[start:end]

    return render_template("index.html",username=username, locations=paginated, page=page, total_pages=total_pages)

#Login route
@app.route("/login", methods = [ "GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        users = load_users()

        if username in users and users[username]==password:
            session["username"] =username
            return redirect(url_for("index")) #redirect to index after loggin
        else:
            error = "Invalid username or password."
    return render_template("login.html", error = error)

#Logout route
@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("index"))

@app.route("/signup",methods=["GET","POST"])
def signup():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirm = request.form.get("confirm")

        users = load_users()

        if username in users:
            error = "Username already in exists."
        elif password != confirm:
            error = "Password do not match."
        else:
            save_user(username, password)
            session["username"] = username
            return redirect(url_for("index"))
    return render_template("signup.html", error=error)

@app.route("/destination/<place>", methods=["GET","POST"] )
def destination_details(place):
    place =unquote(place).lower().strip()
    locations = load_destinations()

    #match ignoring cases and whitespaces
    location = next((loc for loc in locations if loc["place"].lower().strip() == place),None)

    if not location:
        return "Destination not found", 404
    
    reviews = load_reviews()

    #handle review form submission
    if request.method == "POST" and "username" in session:
        comment = request.form.get("comment")
        rating = int(request.form.get("rating",0))
        username = session["username"]

        if comment:
            if location["place"] not in reviews:
                reviews[location["place"]] =[]
            reviews[location["place"]].append({
                "user": username,
                "comment": comment,
                "rating": rating
            })    
            save_reviews(reviews)
            return redirect(url_for("destination_details", place=location["place"].replace(" ", "%20")))

    place_reviews = reviews.get(location["place"], [])

    likes = load_likes()

    #handle likes 
    if request.method == "POST" and "username" in session:
        if "like" in request.form:
            if location["place"] not in likes:
                likes[location["place"]] = {"users":[], "count":0}

            if username in likes[location["place"]]["users"]:
                likes[location["place"]]["users"].remove(username)
                likes[location["place"]]["count"] -= 1
            else:
                likes[location["place"]]["users"].append(username)
                likes[location["place"]]["count"] += 1
            
            save_likes(likes)
            return redirect(url_for("destination_details", place=location["place"].replace(" ", "%20")))
        
    likes=likes.get(location["place"], {"users": [], "count": 0})

    return render_template("destination.html", location = location, reviews=place_reviews, likes=likes)

if __name__=="__main__":
    app.run(debug=True)