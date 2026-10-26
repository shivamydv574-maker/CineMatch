import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CineMatch - Movie Recommendation",
    page_icon="🎬",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f0f0f, #171717, #0b0b0b);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #b8b8b8;
    font-size: 18px;
    margin-bottom: 35px;
}

.movie-card {
    background: linear-gradient(145deg, #242424, #151515);
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 18px;
    border: 1px solid #333333;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.35);
}

.movie-number {
    font-size: 14px;
    color: #aaaaaa;
}

.movie-name {
    font-size: 21px;
    font-weight: 700;
    margin-top: 5px;
}

.similarity {
    color: #00d4ff;
    font-weight: 600;
    margin-top: 8px;
}

.info-box {
    background: #1f1f1f;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #333333;
    text-align: center;
}

.footer {
    text-align: center;
    color: #777777;
    margin-top: 40px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎬 CineMatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Movie Recommendation System</div>',
    unsafe_allow_html=True
)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("tmdb_5000_movies.csv")

    df = df[
        ["title", "genres", "keywords", "overview"]
    ].copy()

    df = df.dropna(subset=["title"])

    return df


df = load_data()

# =========================================================
# CONVERT JSON DATA
# =========================================================

def get_names(value):

    try:

        data = ast.literal_eval(value)

        if isinstance(data, list):

            return " ".join(
                item["name"].replace(" ", "").lower()
                for item in data
                if isinstance(item, dict)
                and "name" in item
            )

    except:

        return ""

    return ""


df["genres"] = (
    df["genres"]
    .fillna("[]")
    .apply(get_names)
)

df["keywords"] = (
    df["keywords"]
    .fillna("[]")
    .apply(get_names)
)

df["overview"] = (
    df["overview"]
    .fillna("")
)

# =========================================================
# CREATE TAGS
# =========================================================

df["tags"] = (
    df["genres"]
    + " "
    + df["keywords"]
    + " "
    + df["overview"]
)

# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

@st.cache_resource
def create_model(tags):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    vectors = vectorizer.fit_transform(tags)

    similarity = cosine_similarity(vectors)

    return similarity


similarity = create_model(
    df["tags"]
)

# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend(movie_name):

    movie_index = df[
        df["title"].str.lower()
        == movie_name.lower()
    ].index

    if len(movie_index) == 0:

        return []

    index = movie_index[0]

    scores = list(
        enumerate(
            similarity[index]
        )
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i, score in scores[1:11]:

        recommendations.append(
            (
                df.iloc[i]["title"],
                score
            )
        )

    return recommendations


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🎬 CineMatch")

    st.write(
        "Select a movie and our ML model "
        "will find similar movies."
    )

    st.divider()

    st.write("### 🧠 Technology")

    st.write("• Python")
    st.write("• Pandas")
    st.write("• Scikit-learn")
    st.write("• TF-IDF")
    st.write("• Cosine Similarity")
    st.write("• Streamlit")

    st.divider()

    st.caption(
        "BCA Data Science & AI Project"
    )

# =========================================================
# MOVIE SELECTION
# =========================================================

st.markdown(
    "### 🔍 Choose Your Favorite Movie"
)

movie_list = sorted(
    df["title"]
    .dropna()
    .unique()
    .tolist()
)

selected_movie = st.selectbox(
    "Select a movie",
    movie_list
)

# =========================================================
# BUTTON
# =========================================================

if st.button(
    "✨ Recommend Movies",
    use_container_width=True
):

    recommendations = recommend(
        selected_movie
    )

    if recommendations:

        st.divider()

        st.markdown(
            f"### 🍿 Movies Similar to **{selected_movie}**"
        )

        # Statistics

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                '<div class="info-box">'
                '🎬<br><b>10</b><br>'
                'Recommendations'
                '</div>',
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                '<div class="info-box">'
                '🧠<br><b>TF-IDF</b><br>'
                'ML Algorithm'
                '</div>',
                unsafe_allow_html=True
            )

        with col3:

            st.markdown(
                '<div class="info-box">'
                '📊<br><b>Cosine</b><br>'
                'Similarity'
                '</div>',
                unsafe_allow_html=True
            )

        st.write("")

        # Movie Cards

        for number, (movie, score) in enumerate(
            recommendations,
            start=1
        ):

            percentage = round(
                score * 100,
                2
            )

            st.markdown(
                f"""
                <div class="movie-card">

                    <div class="movie-number">
                        MOVIE #{number}
                    </div>

                    <div class="movie-name">
                        🎥 {movie}
                    </div>

                    <div class="similarity">
                        ⭐ Similarity: {percentage}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.error(
            "Movie not found in the dataset."
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🎬 CineMatch — Movie Recommendation System<br>
        Built with Python, Machine Learning & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)