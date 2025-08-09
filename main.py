# =========================
# Import Required Libraries
# =========================
import streamlit as st  # Streamlit for creating the web app interface
import pickle           # For loading pre-trained models and objects
import pandas as pd     # For handling data
from sklearn.feature_extraction.text import TfidfVectorizer  # For text feature extraction
from scipy.sparse import hstack, csc_matrix                  # For combining sparse matrices
import matplotlib.pyplot as plt  # For plotting graphs

# =========================
# Function to Load Data and Models
# =========================
@st.cache_data()  # Cache the function output so it's not reloaded every time the app refreshes
def load_model():
    # Load the dataset containing book information
    df = pd.read_pickle("Book_Recommend_Dataset.pickle")

    # Load the trained KNN model
    model = pickle.load(open('knn_model.pickle', 'rb'))
    
    # Load the pre-trained TF-IDF vectorizer
    tfidf = pickle.load(open('tfidf.pickle', 'rb'))
    
    # Load the pre-fitted scaler for numerical features
    scaler = pickle.load(open('scaler.pickle', 'rb'))
    
    return df, model, tfidf, scaler

# =========================
# Load the Data and Models
# =========================
df, model, tfidf, scaler = load_model()

# Extract book titles from the dataset
book_title = df['title'].values

# Create a dictionary mapping book titles to their index positions
title_to_idx = {title: idx for idx, title in enumerate(book_title)}

# =========================
# Streamlit UI
# =========================
st.title("📚 Book Recommendation System")  # Page title

# Dropdown menu to select a book
book_title = st.selectbox("Select a book", book_title)

# =========================
# Generate Recommendations
# =========================
if book_title:
    # Store selection history in session state
    if "history" not in st.session_state:
        st.session_state.history = []

    # Add the selected book to history (avoid duplicates in a row)
    if not st.session_state.history or st.session_state.history[-1] != book_title:
        st.session_state.history.append(book_title)

    # 🎈 Celebration effects
    st.balloons()
    st.snow()

    # Find the index of the selected book
    idx = title_to_idx[book_title]

    # Extract the text description
    text_feature = df.iloc[idx]['text']

    # TF-IDF transformation
    tfidf_feature = tfidf.transform([text_feature])

    # Scale the year feature
    year = scaler.transform([[df.iloc[idx]['year']]])

    # Combine features
    combined = hstack([tfidf_feature, csc_matrix(year)])

    # Find nearest neighbors
    distances, indices = model.kneighbors(combined)

    # =========================
    # Display Recommendations
    # =========================
    st.header("📖 Recommended Books")
    recommendations = []

    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        recommendations.append((i + 1, df.iloc[idx]['title'], dist))

    for rank, title, dist in recommendations:
        st.write(f"{rank}. {title} (Similarity Distance: {dist:.4f})")

    # =========================
    # Show Book Selection History Graph
    # =========================
    st.subheader("📊 Your Book Selection History")

    if st.session_state.history:
        # Create a DataFrame for history
        history_df = pd.DataFrame(st.session_state.history, columns=["Book"])
        
        # Count how many times each book was selected
        count_df = history_df["Book"].value_counts().reset_index()
        count_df.columns = ["Book", "Selections"]

        # Plot using Matplotlib
        fig, ax = plt.subplots()
        ax.barh(count_df["Book"], count_df["Selections"], color="skyblue")
        ax.set_xlabel("Number of Selections")
        ax.set_ylabel("Book Title")
        ax.set_title("Past Book Selections")

        st.pyplot(fig)

   