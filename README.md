# IBM Mini Project

## Overview
This project is a console-based book recommendation system that suggests similar books based on the user’s input. It uses Natural Language Processing (NLP) techniques like TF-IDF vectorization and cosine similarity on the Goodbooks-10k dataset to generate content-based recommendations.

## Features
- Content-based filtering using book titles
- Uses TF-IDF for text vectorization
- Computes cosine similarity between book vectors
- Console-based interface (can be extended to web later)
- Simple and easy to understand Python implementation
- 
## Project Structure
```
AI_Book_Recommender/
├── app.py              # Main file for running the system
├── data/
│   └── books.csv       # Dataset file (from goodbooks-10k)
├── README.md           # Project documentation
```

## Dependencies
Install the required libraries using pip:
```
pip install pandas scikit-learn
```

## How to Run
Clone the repository or download the files.
Place the dataset (books.csv) inside the data/ folder.
Run the project using:
```
python app.py
```
Enter a book title when prompted to get recommendations.

## Dataset
Name: Goodbooks-10k
Contains: 10,000 book titles and user ratings
Format: CSV

## Algorithms Used
TF-IDF (Term Frequency–Inverse Document Frequency): Converts text data into numerical vectors
Cosine Similarity: Measures similarity between book title vectors
Content-Based Filtering: Recommends similar items to the one the user likes

## Future Scope
Add GUI using Streamlit or Flask
Integrate collaborative filtering based on user ratings
Deploy as a web application
Add filtering by author, genre, or publication year
Specify the license under which your project is distributed.

## Contributors
Ananya

Harsh Raj
