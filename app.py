
import streamlit as st
import pandas as pd
import ast
import requests
import os
from pathlib import Path
from urllib.parse import quote_plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PAGE CONFIG

st.set_page_config(
    page_title="CineMatch - Movie Recommendation",
    page_icon="🎬",
    layout="wide"
)

# PROFESSIONAL DESIGN

st.markdown("""
<style>
.stApp {
    background: #0e1117;
    color: white;
}
[data-testid="stSidebar"] {
    background: #171b24;
}
[data-testid="stSidebar"] * {
    color: white;
}
h1, h2, h3, p, label {
    color: white;
}
.main-title {
    text-align: center;
    font-size: 55px;
    font-weight: 800;
    color: #ff4b4b;
}
.subtitle {
    text-align: center;
    color: #aaaaaa;
    font-size: 18px;
    margin-bottom: 35px;
}
.stButton > button {
    border-radius: 10px;
    font-weight: bold;
    width: 100%;
}
[data-testid="stImage"] img {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🎬 CineMatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI Powered Movie Recommendation System</div>',
    unsafe_allow_html=True
)

# TMDB API SETTINGS

try:
    TMDB_CREDENTIAL = st.secrets.get(
        "TMDB_API_KEY", ""
    )
except Exception:
    TMDB_CREDENTIAL = ""

TMDB_CREDENTIAL = str(
    TMDB_CREDENTIAL or os.getenv("TMDB_API_KEY", "")
).strip()

TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

# LOAD DATASET

@st.cache_data
def load_data():
    file_path = Path(__file__).parent / "tmdb_5000_movies.csv"
    movies = pd.read_csv(file_path)

    movies = movies[
        ["id", "title", "overview", "genres",
         "keywords", "vote_average"]
    ].copy()

    movies = movies.dropna(
        subset=["id", "title"]
    ).reset_index(drop=True)

    movies["overview"] = movies["overview"].fillna("")
    movies["genres"] = movies["genres"].fillna("[]")
    movies["keywords"] = movies["keywords"].fillna("[]")

    return movies

movies = load_data()

# CONVERT GENRES AND KEYWORDS

def convert_data(value):
    try:
        data = ast.literal_eval(value)

        if isinstance(data, list):
            return " ".join(
                item.get("name", "")
                for item in data
            )
    except (ValueError, SyntaxError, TypeError):
        pass

    return ""

movies["genres_text"] = movies["genres"].apply(
    convert_data
)

movies["keywords_text"] = movies["keywords"].apply(
    convert_data
)

movies["combined_features"] = (
    movies["title"] + " " +
    movies["overview"] + " " +
    movies["genres_text"] + " " +
    movies["keywords_text"]
)

# MACHINE LEARNING MODEL

@st.cache_resource
def create_model(features):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000
    )

    return vectorizer.fit_transform(features)

feature_matrix = create_model(
    tuple(movies["combined_features"])
)

# TMDB API REQUEST

def tmdb_request(endpoint, extra_params=None):

    if not TMDB_CREDENTIAL:
        return None, "API key not configured"

    url = "https://api.themoviedb.org/3/" + endpoint

    params = {
        "language": "en-US"
    }

    if extra_params:
        params.update(extra_params)

    headers = {}

    if TMDB_CREDENTIAL.startswith("eyJ"):
        headers["Authorization"] = (
            "Bearer " + TMDB_CREDENTIAL
        )
    else:
        params["api_key"] = TMDB_CREDENTIAL

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=(8, 15)
        )

        if response.status_code == 200:
            return response.json(), "200"

        return None, str(response.status_code)

    except requests.exceptions.Timeout:
        return None, "Connection timed out"

    except requests.exceptions.RequestException:
        return None, "Connection failed"

    except ValueError:
        return None, "Invalid API response"

# MOVIE DETAILS, TRAILER AND WATCH PROVIDERS

@st.cache_data(ttl=3600)
def get_movie_details(movie_id):

    data, status = tmdb_request(
        f"movie/{movie_id}",
        {
            "append_to_response": "videos,watch/providers"
        }
    )

    if data is None:
        return None, status

    poster = None

    if data.get("poster_path"):
        poster = (
            TMDB_IMAGE_URL + data["poster_path"]
        )

    # FIND OFFICIAL YOUTUBE TRAILER

    trailer_url = None

    videos = data.get(
        "videos", {}
    ).get("results", [])

    trailers = [
        video for video in videos
        if video.get("site") == "YouTube"
        and video.get("type") == "Trailer"
        and video.get("key")
    ]

    trailers.sort(
        key=lambda video: bool(video.get("official")),
        reverse=True
    )

    if trailers:
        trailer_url = (
            "https://www.youtube.com/watch?v="
            + trailers[0]["key"]
        )

    # LEGAL STREAMING PROVIDERS IN INDIA

    india = data.get(
        "watch/providers", {}
    ).get(
        "results", {}
    ).get("IN", {})

    watch_url = india.get("link")

    provider_names = []

    for category in [
        "flatrate", "free", "ads", "rent", "buy"
    ]:
        for provider in india.get(category, []):
            name = provider.get("provider_name")

            if name and name not in provider_names:
                provider_names.append(name)

    return {
        "poster": poster,
        "rating": data.get("vote_average"),
        "overview": data.get("overview"),
        "trailer": trailer_url,
        "watch_url": watch_url,
        "providers": provider_names
    }, status

# RECOMMEND MOVIES

def recommend_movies(movie_name):

    matches = movies.index[
        movies["title"] == movie_name
    ].tolist()

    if not matches:
        return []

    movie_index = matches[0]

    scores = cosine_similarity(
        feature_matrix[movie_index],
        feature_matrix
    ).flatten()

    ranked = scores.argsort()[::-1]

    recommendations = []

    for index in ranked:

        if index == movie_index:
            continue

        movie = movies.iloc[index]

        recommendations.append({
            "id": int(movie["id"]),
            "title": movie["title"],
            "overview": movie["overview"],
            "rating": movie["vote_average"],
            "score": float(scores[index])
        })

        if len(recommendations) == 10:
            break

    return recommendations

# SIDEBAR

st.sidebar.title("🎬 CineMatch")

st.sidebar.info(
    "Discover movies, watch trailers "
    "and find legal streaming platforms."
)

if TMDB_CREDENTIAL:
    st.sidebar.success("🔑 TMDB API configured")
else:
    st.sidebar.warning("TMDB API not configured")

if st.sidebar.button("🔧 Test TMDB API"):

    data, status = tmdb_request("configuration")

    if status == "200":
        st.sidebar.success("✅ TMDB API working!")
    else:
        st.sidebar.warning(
            f"TMDB Status: {status}"
        )

st.sidebar.markdown("---")

st.sidebar.markdown("""
### 🧠 Technology

- Python
- Pandas
- Streamlit
- Machine Learning
- TF-IDF
- Cosine Similarity
- TMDB API
""")

# MOVIE SEARCH

st.subheader("🔎 Find Your Next Movie")

movie_list = sorted(
    movies["title"].unique()
)

selected_movie = st.selectbox(
    "Select a movie",
    movie_list
)

if "selected_result" not in st.session_state:
    st.session_state.selected_result = None

if st.button(
    "🎯 Recommend Movies",
    type="primary"
):
    st.session_state.selected_result = selected_movie

# DISPLAY RECOMMENDATIONS

if st.session_state.selected_result:

    result_movie = st.session_state.selected_result

    recommendations = recommend_movies(
        result_movie
    )

    st.success(
        f"🍿 Movies similar to {result_movie}"
    )

    for row_start in range(
        0, len(recommendations), 5
    ):

        row_movies = recommendations[
            row_start:row_start + 5
        ]

        columns = st.columns(5)

        for col, movie in zip(columns, row_movies):

            with col:

                details, status = get_movie_details(
                    movie["id"]
                )

                details = details or {}

                poster = details.get("poster")

                # MOVIE POSTER

                if poster:
                    st.image(
                        poster,
                        use_container_width=True
                    )
                else:
                    st.markdown("""
<div style="height:260px;background:#20242b;border-radius:12px;display:flex;align-items:center;justify-content:center;text-align:center;color:#aaa;">
🎬 Poster Not Available
</div>
""", unsafe_allow_html=True)

                # MOVIE TITLE

                st.markdown(
                    f"**{movie['title']}**"
                )

                # RATING

                rating = details.get("rating")

                if not rating:
                    rating = movie["rating"]

                if pd.notna(rating):
                    st.write(
                        f"⭐ Rating: {rating:.1f}/10"
                    )

                st.caption(
                    f"Similarity: {movie['score'] * 100:.1f}%"
                )

                # WATCH TRAILER BUTTON

                trailer_url = details.get("trailer")

                if trailer_url:

                    st.link_button(
                        "▶️ Watch Trailer",
                        trailer_url,
                        use_container_width=True
                    )

                else:

                    search_url = (
                        "https://www.youtube.com/results?"
                        "search_query="
                        + quote_plus(
                            movie["title"] + " official trailer"
                        )
                    )

                    st.link_button(
                        "🔎 Find Trailer",
                        search_url,
                        use_container_width=True
                    )

                # WATCH NOW BUTTON

                watch_url = details.get("watch_url")

                if watch_url:

                    st.link_button(
                        "🍿 Watch Now",
                        watch_url,
                        use_container_width=True
                    )

                    providers = details.get(
                        "providers", []
                    )

                    if providers:
                        st.caption(
                            "Available options: "
                            + ", ".join(providers)
                        )

                else:

                    movie_url = (
                        "https://www.themoviedb.org/movie/"
                        + str(movie["id"])
                        + "/watch?locale=IN"
                    )

                    st.link_button(
                        "🍿 Where to Watch",
                        movie_url,
                        use_container_width=True
                    )

                # MOVIE DETAILS

                overview = (
                    details.get("overview")
                    or movie["overview"]
                )

                with st.expander("📖 Movie Details"):

                    st.write(
                        overview or "Details unavailable."
                    )

                    if not watch_url:
                        st.caption(
                            "Streaming availability in India "
                            "could not be confirmed."
                        )

# FOOTER

st.markdown("---")

st.markdown("""
<div style="text-align:center;color:#aaa;padding:20px;">
🎬 CineMatch | AI Powered Movie Recommendation System<br><br>
Built with Python, Streamlit, Machine Learning and TMDB API<br><br>
This product uses the TMDB API but is not endorsed or certified by TMDB.
<br>
Streaming availability data provided by JustWatch.
</div>
""", unsafe_allow_html=True)

st.image(
    "https://www.themoviedb.org/assets/2/v4/logos/v2/blue_short-0e2e5c6c7c10a14bdf99a6d2fce57f4f.svg",
    width=120
)
