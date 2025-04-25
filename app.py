import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ----------------------------
# 1. Load and Prepare Dataset
# ----------------------------

def load_and_prepare_data(filepath):
    try:
        books_df = pd.read_csv(filepath)
        books_df = books_df.dropna(subset=['title'])  # Ensure there are no missing titles
        books_df['lower_title'] = books_df['title'].str.lower()  # Normalize titles
        books_df = books_df.drop_duplicates(subset='lower_title')  # Remove duplicate titles
        return books_df
    except Exception as e:
        print("Error loading data:", e)
        return None

# ----------------------------
# 2. Preprocessing Titles
# ----------------------------

def preprocess_text(text):
    return re.sub(r'[^a-z\s]', '', str(text).lower())

def preprocess_titles(books_df):
    books_df['processed_title'] = books_df['title'].apply(preprocess_text)
    return books_df

# ----------------------------
# 3. Model Creation
# ----------------------------

def build_model(processed_texts, n_neighbors=6):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(processed_texts)

    model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    model.fit(tfidf_matrix)

    return model, tfidf_matrix

# ----------------------------
# 4. Recommendation Logic
# ----------------------------

def recommend_books(title, books_df, title_to_index, model, tfidf_matrix):
    title = title.lower()

    if title not in title_to_index:
        return None

    idx = title_to_index[title]
    distances, indices = model.kneighbors(tfidf_matrix[idx])

    # Skip the first book (it’s the same as input)
    recommendations = books_df.iloc[indices[0][1:]]['title'].tolist()
    return recommendations

# ----------------------------
# 5. Main Loop (User Interaction)
# ----------------------------

def main():
    print("📚 Welcome to the Book Recommendation System 📚\n")

    # Step 1: Load and preprocess
    books_df = load_and_prepare_data("data/books.csv")
    if books_df is None:
        return

    books_df = preprocess_titles(books_df)

    # Step 2: Build title mapping and model
    title_to_index = pd.Series(books_df.index, index=books_df['lower_title'])
    model, tfidf_matrix = build_model(books_df['processed_title'])

    # Step 3: User Interaction
    while True:
        book_title = input("🔎 Enter the book title (Type 'end' to quit): ").strip()
        if book_title.lower() == 'end':
            print("👋 Exiting the system. Goodbye!")
            break

        recommendations = recommend_books(book_title, books_df, title_to_index, model, tfidf_matrix)

        if recommendations is None:
            print("❌ Sorry! Book not found. Try another title.\n")
        else:
            print(f"\n📖 Top 5 recommendations for '{book_title}':\n")
            for i, book in enumerate(recommendations, 1):
                print(f"{i}. {book}")
            print("\n")

# ----------------------------
# Run the application
# ----------------------------

if __name__ == "__main__":
    main()
