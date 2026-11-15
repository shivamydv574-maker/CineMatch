import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CineMatch - Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #aaaaaa;
    font-size: 18px;
    margin-bottom: 30px;
}

.movie-title {
    font-size: 20px;
    font-weight: 700;
    margin-top: 10px;
}

.movie-info {
    color: #bbbbbb;
    font-size: 14px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🎬 CineMatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI Powered Movie Recommendation System</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    movies = pd.read_csv("tmdb_5000_movies.csv")

    required_columns = [
        "id",
        "title",
        "overview",
        "genres",
        "keywords",
        "vote_average"
    ]

    movies = movies[required_columns]

    movies["overview"] = movies["overview"].fillna("")
    movies["genres"] = movies["genres"].fillna("[]")
    movies["keywords"] = movies["keywords"].fillna("")

    return movies


movies = load_data()


# =========================================================
# GENRES
# =========================================================

def convert_genres(value):

    try:

        data = ast.literal_eval(value)

        if isinstance(data, list):

            return " ".join(
                item.get("name", "")
                for item in data
            )

        return ""

    except:

        return ""


movies["genres_text"] = movies["genres"].apply(
    convert_genres
)


# =========================================================
# KEYWORDS
# =========================================================

def convert_keywords(value):

    try:

        data = ast.literal_eval(value)

        if isinstance(data, list):

            return " ".join(
                item.get("name", "")
                for item in data
            )

        return ""

    except:

        return ""


movies["keywords_text"] = movies["keywords"].apply(
    convert_keywords
)


# =========================================================
# COMBINE FEATURES
# =========================================================

movies["combined_features"] = (
    movies["title"].fillna("") + " " +
    movies["overview"].fillna("") + " " +
    movies["genres_text"].fillna("") + " " +
    movies["keywords_text"].fillna("")
)


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

@st.cache_resource
def create_model(data):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000
    )

    feature_matrix = vectorizer.fit_transform(
        data["combined_features"]
    )

    similarity_matrix = cosine_similarity(
        feature_matrix
    )

    return similarity_matrix


similarity = create_model(movies)


# =========================================================
# TMDB SETTINGS
# =========================================================

TMDB_CREDENTIAL = st.secrets.get(
    "TMDB_API_KEY",
    ""
)

TMDB_IMAGE_URL = (
    "https://image.tmdb.org/t/p/w500"
)


# =========================================================
# TMDB MOVIE DETAILS
# =========================================================

@st.cache_data(ttl=3600)
def get_movie_details(movie_title):

    # If API credential is missing
    if not TMDB_CREDENTIAL:

        return {
            "poster": None,
            "rating": None,
            "overview": None
        }

    url = (
        "https://api.themoviedb.org/3/search/movie"
    )

    params = {
        "query": movie_title,
        "language": "en-US"
    }

    headers = {}

    # -----------------------------------------------------
    # API READ ACCESS TOKEN
    # -----------------------------------------------------

    if TMDB_CREDENTIAL.startswith("eyJ"):

        headers["Authorization"] = (
            "Bearer " + TMDB_CREDENTIAL
        )

    # -----------------------------------------------------
    # API KEY
    # -----------------------------------------------------

    else:

        params["api_key"] = TMDB_CREDENTIAL

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=5
        )

        # -------------------------------------------------
        # If TMDB gives error
        # -------------------------------------------------

        if response.status_code != 200:

            return {
                "poster": None,
                "rating": None,
                "overview": None
            }

        data = response.json()

        results = data.get(
            "results",
            []
        )

        if not results:

            return {
                "poster": None,
                "rating": None,
                "overview": None
            }

        movie = results[0]

        # -------------------------------------------------
        # POSTER
        # -------------------------------------------------

        poster_path = movie.get(
            "poster_path"
        )

        poster = None

        if poster_path:

            poster = (
                TMDB_IMAGE_URL +
                poster_path
            )

        # -------------------------------------------------
        # RETURN DETAILS
        # -------------------------------------------------

        return {
            "poster": poster,
            "rating": movie.get(
                "vote_average"
            ),
            "overview": movie.get(
                "overview"
            )
        }

    # -----------------------------------------------------
    # INTERNET / TIMEOUT ERROR
    # -----------------------------------------------------

    except requests.exceptions.Timeout:

        return {
            "poster": None,
            "rating": None,
            "overview": None
        }

    except requests.exceptions.ConnectionError:

        return {
            "poster": None,
            "rating": None,
            "overview": None
        }

    except Exception:

        return {
            "poster": None,
            "rating": None,
            "overview": None
        }


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend_movies(movie_name):

    movie_name = movie_name.lower().strip()

    matches = movies[
        movies["title"]
        .str.lower()
        .str.contains(
            movie_name,
            na=False
        )
    ]

    if matches.empty:

        return None

    movie_index = matches.index[0]

    scores = list(
        enumerate(
            similarity[movie_index]
        )
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in scores[1:11]:

        recommendations.append({

            "title":
                movies.iloc[index]["title"],

            "score":
                float(score),

            "overview":
                movies.iloc[index]["overview"],

            "rating":
                movies.iloc[index]["vote_average"]

        })

    return recommendations


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎬 CineMatch")

st.sidebar.markdown(
    "### Movie Recommendation System"
)

st.sidebar.info(
    "Enter a movie name and CineMatch "
    "will recommend similar movies."
)


# =========================================================
# TMDB STATUS
# =========================================================

if TMDB_CREDENTIAL:

    st.sidebar.success(
        "🔑 TMDB API configured"
    )

else:

    st.sidebar.warning(
        "⚠️ TMDB API not configured"
    )


# =========================================================
# TEST TMDB API
# =========================================================

if st.sidebar.button(
    "🔧 Test TMDB API"
):

    if not TMDB_CREDENTIAL:

        st.sidebar.error(
            "TMDB API credential not found!"
        )

    else:

        test_url = (
            "https://api.themoviedb.org/3/search/movie"
        )

        test_params = {
            "query": "Avatar"
        }

        test_headers = {}

        if TMDB_CREDENTIAL.startswith("eyJ"):

            test_headers["Authorization"] = (
                "Bearer " + TMDB_CREDENTIAL
            )

        else:

            test_params["api_key"] = (
                TMDB_CREDENTIAL
            )

        try:

            test_response = requests.get(
                test_url,
                params=test_params,
                headers=test_headers,
                timeout=5
            )

            st.sidebar.write(
                "TMDB Status:",
                test_response.status_code
            )

            if test_response.status_code == 200:

                st.sidebar.success(
                    "✅ TMDB API is working!"
                )

            elif test_response.status_code == 401:

                st.sidebar.error(
                    "❌ TMDB authentication failed."
                )

            else:

                st.sidebar.warning(
                    "⚠️ TMDB returned an error."
                )

        except requests.exceptions.Timeout:

            st.sidebar.warning(
                "⚠️ TMDB connection timed out."
            )

        except requests.exceptions.ConnectionError:

            st.sidebar.warning(
                "⚠️ Cannot connect to TMDB."
            )

        except Exception as e:

            st.sidebar.warning(
                "TMDB test failed."
            )


# =========================================================
# SEARCH SECTION
# =========================================================

st.subheader(
    "🔎 Find Your Next Movie"
)


movie_list = sorted(
    movies["title"]
    .dropna()
    .unique()
)


selected_movie = st.selectbox(
    "Select a movie",
    movie_list
)


recommend_button = st.button(
    "🎯 Recommend Movies",
    type="primary"
)


# =========================================================
# RECOMMENDATIONS
# =========================================================

if recommend_button:

    recommendations = recommend_movies(
        selected_movie
    )

    if recommendations is None:

        st.error(
            "❌ Movie not found."
        )

    else:

        st.success(
            f"Movies similar to **{selected_movie}**"
        )

        # -------------------------------------------------
        # SHOW 5 MOVIES PER ROW
        # -------------------------------------------------

        for row_start in range(
            0,
            len(recommendations),
            5
        ):

            row_movies = recommendations[
                row_start:row_start + 5
            ]

            columns = st.columns(5)

            for col, movie in zip(
                columns,
                row_movies
            ):

                with col:

                    # -------------------------------------
                    # GET TMDB DETAILS
                    # -------------------------------------

                    details = get_movie_details(
                        movie["title"]
                    )

                    poster = details.get(
                        "poster"
                    )

                    # -------------------------------------
                    # POSTER
                    # -------------------------------------

                    if poster:

                        st.image(
                            poster,
                            use_container_width=True
                        )

                    else:

                        st.markdown(
                            """
                            <div style="
                                height:300px;
                                display:flex;
                                flex-direction:column;
                                align-items:center;
                                justify-content:center;
                                background:#20242b;
                                border-radius:10px;
                                color:#999;
                                font-size:18px;
                                text-align:center;
                            ">
                                <div style="
                                    font-size:55px;
                                    margin-bottom:10px;
                                ">
                                    🎬
                                </div>

                                Poster Not Available

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # -------------------------------------
                    # TITLE
                    # -------------------------------------

                    st.markdown(
                        f"""
                        <div class="movie-title">
                            {movie["title"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # -------------------------------------
                    # RATING
                    # -------------------------------------

                    tmdb_rating = details.get(
                        "rating"
                    )

                    if (
                        tmdb_rating is not None
                        and tmdb_rating > 0
                    ):

                        st.markdown(
                            f"""
                            ⭐ TMDB Rating:
                            **{tmdb_rating:.1f}/10**
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        dataset_rating = movie[
                            "rating"
                        ]

                        if pd.notna(
                            dataset_rating
                        ):

                            st.markdown(
                                f"""
                                ⭐ Rating:
                                **{dataset_rating:.1f}/10**
                                """,
                                unsafe_allow_html=True
                            )

                    # -------------------------------------
                    # SIMILARITY
                    # -------------------------------------

                    st.caption(
                        "🤖 Similarity: "
                        f"{movie['score'] * 100:.1f}%"
                    )

                    # -------------------------------------
                    # DETAILS
                    # -------------------------------------

                    overview = details.get(
                        "overview"
                    )

                    if not overview:

                        overview = movie[
                            "overview"
                        ]

                    if overview:

                        with st.expander(
                            "📖 Details"
                        ):

                            st.write(
                                overview
                            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#888;
        padding:10px;
    ">
        🎬 CineMatch | AI Powered Movie Recommendation System
        <br><br>
        Built with Python, Streamlit,
        Machine Learning & TMDB API
    </div>
    """,
    unsafe_allow_html=True
)
