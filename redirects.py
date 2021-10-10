### IMPORT LIBRARIES ###

import streamlit as st
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', None)
import seaborn as sns
cm = sns.light_palette("red", as_cmap=True)
st.set_page_config(layout="wide")
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from stqdm import stqdm


def url_preprocessing(url):
    url.split("/")[-1]

def best_strings_similarity(x):
    urls = ['url', 'Url', 'URL', 'page', 'Page', 'PAGE']

    choices = new_file[select_new_column].tolist()

    if select_old_column in urls:

        if checkbox_whole_old_url:
            return process.extract(x, choices, scorer=fuzz.token_sort_ratio)
        else:
            if ".html" in x:
                x = x.split("/")[-1].lower()
                return process.extract(x, choices, scorer=fuzz.token_sort_ratio)
            else:
                x = x.split("/")[-2].lower()
                return process.extract(x, choices, scorer=fuzz.token_sort_ratio)
    else:
        return process.extract(x, choices, scorer=fuzz.token_sort_ratio)


st.image("magicien.png", width=100)
st.title("Redirects Wizard")

st.sidebar.header("Setup")

algorithm_selectbox = st.sidebar.radio(
    "1. Select the matching algorithm",
    ("Strings similarity", "BM25F (inactive)", "Semantic matching (inactive)")
)

old_urls_upload = st.sidebar.file_uploader(label="2. Upload Your Old URLs", type=["csv"], help="CSV file with \";\" separator")

if old_urls_upload:
    old_file = pd.read_csv(old_urls_upload, encoding='utf-8', sep=";")
    old_file_colnames = old_file.columns.tolist()
    select_old_column = st.sidebar.radio("Which columns to match?", old_file_colnames)
    checkbox_whole_old_url = st.sidebar.checkbox("Use the whole URL to matching", help = "By default, we will match the last URL elements.")
    st.write(len(old_file), " old urls successfully added")
    st.dataframe(old_file[:10].style.highlight_null(null_color='#E7E7E7'))
else:
    old_file = "old_file_no_added"

new_urls_upload = st.sidebar.file_uploader(label="3. Upload Your New URLs", type=["csv"], help="CSV file with \";\" separator")

if new_urls_upload:
    new_file = pd.read_csv(new_urls_upload, encoding='utf-8', sep=";")
    new_file_colnames = new_file.columns.tolist()
    select_new_column = st.sidebar.radio("Which columns to match?", new_file_colnames)
    if checkbox_whole_old_url:
        checkbox_whole_new_url = st.sidebar.checkbox("Use the whole URL", help = "By default, we will match the last URL elements.", value=True)
    else:
        checkbox_whole_new_url = st.sidebar.checkbox("Use the whole URL", help = "By default, we will match the last URL elements.", value=False)

    st.write(len(new_file), " new urls successfully added")
    st.dataframe(new_file[:10].style.highlight_null(null_color='#E7E7E7'))




### DISPLAY RESULT ###

if new_urls_upload and old_urls_upload:
    match_button = st.button('Match!')
    if match_button:
        st.header('Result :')

        with st.spinner('Wait, magic happens ...'):
            stqdm.pandas()

            old_file['New Url'] = old_file[select_old_column].progress_map(lambda x: best_strings_similarity(x)[0])
            old_file[['New Item', 'Similarity Score']] = pd.DataFrame(old_file['New Url'].tolist(), index=old_file.index)
            redirects_plan = old_file[[select_old_column, 'New Item', 'Similarity Score']]

            st.success('All items have been successfully matched!')

            by_simscore = redirects_plan.groupby("Similarity Score").count()
            st.bar_chart(by_simscore)

            st.dataframe(redirects_plan.style.background_gradient(cmap='flare', subset=['Similarity Score']))



        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(redirects_plan)

        st.download_button(
            label = "Download data as CSV",
            data = csv,
            file_name = 'redirects_plan.csv',
            mime = 'text/csv',
        )
