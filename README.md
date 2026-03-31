# 🎬 Movie Recommendation System (Apriori + Flask)

An intelligent movie recommendation system built using the Apriori algorithm and deployed with a Flask web interface.

---

## 🚀 Live Features

* 🔍 Search for any movie
* 🎯 Get personalized recommendations
* 📊 Based on association rule mining (Apriori)
* 🖼️ Movie posters + confidence scores
* 🎨 Clean UI using Tailwind CSS

---

## 🧠 How It Works

This system uses **Association Rule Mining (Apriori Algorithm)**:

1. Movie watch history is treated as transactions
2. Frequent itemsets are generated
3. Association rules are created
4. Recommendations are made based on:

   * Confidence score
   * Genre filtering

---

## 🛠️ Tech Stack

* Python 🐍
* Flask 🌐
* Pandas
* mlxtend (Apriori Algorithm)
* Tailwind CSS 🎨

---

## 📂 Project Structure

```
movie-recommender/
 ├── app.py
 ├── templates/
 │     └── index.html
 ├── requirements.txt
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

---

## 📸 Preview

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/52f0decf-0e71-4648-b528-6e9ed4d63ce9" />


---

## ✨ Key Highlights

* Implements **Market Basket Analysis** for movies
* Handles **typo correction** using difflib
* Filters recommendations based on **genre similarity**
* Displays **confidence-based ranking**

---

## 🔮 Future Improvements

* Use real datasets (Netflix / IMDb)
* Add collaborative filtering
* Deploy using Render / Railway
* Add user login system

---

## 👩‍💻 Author

Namratha N

---

⭐ If you like this project, give it a star!
