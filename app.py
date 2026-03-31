from flask import Flask, render_template, request
import pandas as pd
import difflib
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

app = Flask(__name__)

# -------------------------
# MOVIE TRANSACTIONS DATASET
# -------------------------

transactions = [

# Marvel
['Avengers','Iron Man','Thor','Captain America'],
['Avengers','Spider-Man','Iron Man'],
['Spider-Man','Doctor Strange','Avengers'],
['Thor','Avengers','Iron Man'],

# Horror
['The Conjuring','Annabelle','Insidious'],
['Insidious','Annabelle'],
['The Conjuring','Insidious'],

# Romance
['Titanic','The Notebook','La La Land'],
['Titanic','La La Land'],
['The Notebook','La La Land'],

# Bollywood
['3 Idiots','Dil Chahta Hai','Zindagi Na Milegi Dobara'],
['3 Idiots','Zindagi Na Milegi Dobara'],
['Dil Chahta Hai','Zindagi Na Milegi Dobara'],

# Kannada
['KGF','Kantara','Charlie 777'],
['Kantara','Charlie 777'],
['KGF','Kantara'],

# Extra
['Interstellar','Inception','The Dark Knight'],
['Interstellar','The Dark Knight'],
['Inception','The Dark Knight'],

['Dangal','3 Idiots'],
['Dangal','Lagaan'],
['Lagaan','3 Idiots'],

['Kabir Singh','Tamasha'],
['Tamasha','Yeh Jawaani Hai Deewani'],
['Yeh Jawaani Hai Deewani','Dil Chahta Hai'],

# More variety
['Baahubali','KGF'],
['RRR','Baahubali'],
['Kantara','Baahubali'],
['RRR','KGF']
]

# -------------------------
# GENRES
# -------------------------

genres = {

"KGF":"Action",
"Kantara":"Action",
"Charlie 777":"Drama",
"Baahubali":"Action",
"RRR":"Action",

"3 Idiots":"Drama",
"Dangal":"Drama",
"Lagaan":"Drama",

"Titanic":"Romance",
"The Notebook":"Romance",
"La La Land":"Romance",

"Avengers":"Superhero",
"Iron Man":"Superhero",
"Thor":"Superhero",
"Spider-Man":"Superhero",
"Captain America":"Superhero",

"The Conjuring":"Horror",
"Annabelle":"Horror",
"Insidious":"Horror",

"Interstellar":"Sci-Fi",
"Inception":"Sci-Fi",
"The Dark Knight":"Action",

"Dil Chahta Hai":"Drama",
"Zindagi Na Milegi Dobara":"Drama",
"Yeh Jawaani Hai Deewani":"Romance",

"Kabir Singh":"Romance",
"Tamasha":"Romance"
}

# -------------------------
# MOVIE POSTERS
# -------------------------

posters = {

"Avengers":"https://image.tmdb.org/t/p/w500/RYMX2wcKCBAr24UyPD7xwmjaTn.jpg",
"Iron Man":"https://image.tmdb.org/t/p/w500/78lPtwv72eTNqFW9COBYI0dWDJa.jpg",
"Thor":"https://image.tmdb.org/t/p/w500/prSfAi1xGrhLQNxVSUFh61xQ4Qy.jpg",
"Captain America":"https://image.tmdb.org/t/p/w500/vSNxAJTlD0r02V9sPYpOjqDZXUK.jpg",
"Spider-Man":"https://image.tmdb.org/t/p/w500/gh4cZbhZxyTbgxQPxD0dOudNPTn.jpg",

"The Conjuring":"https://image.tmdb.org/t/p/w500/wVYREutTvI2tmxr6ujrHT704wGF.jpg",
"Annabelle":"https://image.tmdb.org/t/p/w500/yAgI51QvUgni6CqH0nQxV8H7lV.jpg",
"Insidious":"https://image.tmdb.org/t/p/w500/nRPoI6oE6ZqS3Q9C1Qn6zkRgZN.jpg",

"Titanic":"https://image.tmdb.org/t/p/w500/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg",
"The Notebook":"https://image.tmdb.org/t/p/w500/rNzQyW4f8B8cQeg7Dgj3n8MZQyC.jpg",
"La La Land":"https://image.tmdb.org/t/p/w500/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg",

"3 Idiots":"https://image.tmdb.org/t/p/w500/66A9MqXOyVFCssoloscw79z8Tew.jpg",
"Dil Chahta Hai":"https://image.tmdb.org/t/p/w500/3qjY1n2p0v5Y8ZndKyVKoU9pYNC.jpg",
"Zindagi Na Milegi Dobara":"https://image.tmdb.org/t/p/w500/7Y5V7cB8O6jwxKF1uHmIiiaivGi.jpg",

"KGF":"https://image.tmdb.org/t/p/w500/ltHlJwvxKv7d0ooCiKSAvfwV9tX.jpg",
"Kantara":"https://image.tmdb.org/t/p/w500/6Kk7wYJYqRR3YKHyCuXapnwXCfJ.jpg",
"Charlie 777":"https://image.tmdb.org/t/p/w500/9zYx0M0RKZ77N8BINU3CFAaj6Mq.jpg",

"Interstellar":"https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
"Inception":"https://image.tmdb.org/t/p/w500/qmDpIHrmpJINaRKAfWQfftjCdyi.jpg",
"The Dark Knight":"https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg",

"Dangal":"https://image.tmdb.org/t/p/w500/p2lVAcPuRPSO8Al6hDDGwF0L3gE.jpg",
"Lagaan":"https://image.tmdb.org/t/p/w500/6xKCYgH16UuwEGAyroLU6p8HLIn.jpg",

"Kabir Singh":"https://image.tmdb.org/t/p/w500/7WsyChQLEftFiDOVTGkv3hFpyyt.jpg",
"Tamasha":"https://image.tmdb.org/t/p/w500/A1t8LojRxurx8WcN6iYG3PaY5E9.jpg",
"Yeh Jawaani Hai Deewani":"https://image.tmdb.org/t/p/w500/em39H81XLCDgXsI7YcC1ajqvUqk.jpg",

"Baahubali":"https://image.tmdb.org/t/p/w500/9BAjt8nSSms62uOVYn1t3C3dVto.jpg",
"RRR":"https://image.tmdb.org/t/p/w500/nEufeZlyAOLqO2brrs0yeF1lgXO.jpg"
}

# -------------------------
# APRIORI MODEL
# -------------------------

te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)

df = pd.DataFrame(te_data, columns=te.columns_)

frequent = apriori(df, min_support=0.05, use_colnames=True)
rules = association_rules(frequent, metric="confidence", min_threshold=0.1)

# -------------------------
# FLASK ROUTE
# -------------------------

@app.route("/", methods=["GET","POST"])
def home():

    recs=[]
    movie=""

    if request.method == "POST":

        movie = request.form["movie"]

        matches = difflib.get_close_matches(movie, df.columns, n=1)

        if matches:
            movie = matches[0]

        movie_genre = genres.get(movie,"")

        for _,row in rules.iterrows():

            if movie in list(row["antecedents"]):

                for m in list(row["consequents"]):

                    if genres.get(m,"") == movie_genre and m != movie:

                        if not any(r["title"] == m for r in recs):

                            recs.append({
                                "title":m,
                                "confidence":round(row["confidence"]*100,1),
                                "poster":posters.get(m,"")
                            })

        # sort by confidence
        recs = sorted(recs, key=lambda x: x["confidence"], reverse=True)

        # ensure max 5
        recs = recs[:5]

        # fallback if less than 5
        if len(recs) < 5:

            for m in posters:

                if genres.get(m,"") == movie_genre and m != movie:

                    if not any(r["title"] == m for r in recs):

                        recs.append({
                            "title":m,
                            "confidence":50,
                            "poster":posters.get(m,"")
                        })

                if len(recs) >=5:
                    break

    return render_template("index.html",movie=movie,recs=recs)


if __name__=="__main__":
    app.run(debug=True)