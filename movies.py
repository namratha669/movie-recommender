import pandas as pd
import difflib
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Movie watching transactions
transactions = [
    ['Avengers', 'Iron Man', 'Thor'],
    ['Avengers', 'Captain America', 'Iron Man'],
    ['The Conjuring', 'Annabelle'],
    ['Titanic', 'The Notebook'],
    ['3 Idiots', 'Zindagi Na Milegi Dobara'],
    ['KGF', 'Kantara'],
    ['Avengers', 'Spider-Man', 'Iron Man'],
    ['The Conjuring', 'Insidious'],
    ['Titanic', 'La La Land'],
    ['3 Idiots', 'Dil Chahta Hai'],
    ['KGF', 'Charlie 777'],
    ['Spider-Man', 'Doctor Strange'],
    ['Annabelle', 'Insidious'],
    ['The Notebook', 'La La Land'],
    ['Zindagi Na Milegi Dobara', 'Dil Chahta Hai'],
    ['Kantara', 'Charlie 777'],
    ['Avengers', 'Thor'],
    ['Titanic', 'The Notebook', 'La La Land'],
    ['3 Idiots', 'Zindagi Na Milegi Dobara', 'Dil Chahta Hai'],
    ['KGF', 'Kantara', 'Charlie 777']
]

# Genre dictionary
movie_genres = {
    "Avengers": "Action / Superhero",
    "Iron Man": "Action / Superhero",
    "Thor": "Action / Superhero",
    "Captain America": "Action / Superhero",
    "Spider-Man": "Action / Superhero",
    "Doctor Strange": "Action / Fantasy",

    "The Conjuring": "Horror",
    "Annabelle": "Horror",
    "Insidious": "Horror",

    "Titanic": "Romance / Drama",
    "The Notebook": "Romance / Drama",
    "La La Land": "Romance / Musical",

    "3 Idiots": "Comedy / Drama / Bollywood",
    "Zindagi Na Milegi Dobara": "Drama / Bollywood",
    "Dil Chahta Hai": "Drama / Bollywood",

    "KGF": "Action / Kannada",
    "Kantara": "Action / Mythological / Kannada",
    "Charlie 777": "Drama / Kannada"
}

print("Movie Transactions:")
print(transactions)

# Convert to one-hot encoding
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)

df_movies = pd.DataFrame(te_data, columns=te.columns_)

print("\nEncoded Dataset:")
print(df_movies)

# Apply Apriori
frequent_itemsets = apriori(df_movies, min_support=0.1, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)

print("\n🎬 Movie Recommendation System")

# All movie names
all_movies = list(df_movies.columns)

# User input
movie_input = input("Enter a movie you like: ")

# Smart suggestion if spelling is wrong
matches = difflib.get_close_matches(movie_input, all_movies, n=1, cutoff=0.6)

if matches:
    movie_input = matches[0]
else:
    print("Movie not found in dataset.")
    exit()

print(f"\nYou watched: {movie_input}")

# Show genre
genre = movie_genres.get(movie_input, "Unknown Genre")
print(f"Genre: {genre}")

recommendations = {}

for index, row in rules.iterrows():
    antecedent = list(row['antecedents'])

    if movie_input in antecedent:
        movie = list(row['consequents'])[0]
        confidence = row['confidence']

        if movie not in recommendations or confidence > recommendations[movie]:
            recommendations[movie] = confidence

# Sort recommendations by confidence
recommendations = dict(sorted(recommendations.items(), key=lambda x: x[1], reverse=True))

if recommendations:
    print("\nBecause you watched", movie_input, "you might also like:\n")

    for i, (movie, conf) in enumerate(recommendations.items(), 1):
        genre = movie_genres.get(movie, "Unknown Genre")
        print(f"{i}. {movie} - {genre} ({conf*100:.1f}% confidence)")
else:
    print("No recommendations found.")

print("\nAssociation Rules Generated:")
print(rules[['antecedents','consequents','support','confidence']])