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
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

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

def alg_strings_similarity():
    with st.spinner('Wait, magic happens ...'):
        stqdm.pandas()

        old_file['New Url'] = old_file[select_old_column].progress_map(lambda x: best_strings_similarity(x)[0])
        old_file[['New Item', 'Similarity Score']] = pd.DataFrame(old_file['New Url'].tolist(), index=old_file.index)
        redirects_plan = old_file[[select_old_column, 'New Item', 'Similarity Score']]

        st.success('All items have been successfully matched!')

        # by_simscore = redirects_plan.groupby("Similarity Score").count()
        # st.bar_chart(by_simscore)

        st.dataframe(redirects_plan.style.background_gradient(cmap='flare', subset=['Similarity Score']))

    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(redirects_plan)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='redirects_plan.csv',
        mime='text/csv',
    )


def bert(x, model):
    embeddings1 = model.encode(x, convert_to_tensor=True)
    embeddings2 = model.encode(new_file[select_new_column], convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    d = dict(enumerate(cosine_scores.flatten(), 1))

    #{el: 0 for el in new_file['Url']}
    dictionary = dict(zip(new_file.iloc[:, 0], d.values()))

    dict2 = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    match_key = list(dict2.keys())[0]
    match_values = list(dict2.values())[0]
    result = f"{match_key}, {match_values}"
    # st.write(match_key)
    return result



def alg_semantic_best_match(x, model):
    embeddings1 = model.encode(x, convert_to_tensor=True)
    embeddings2 = model.encode(new_file['H1'], convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    d = dict(enumerate(cosine_scores.flatten(), 1))

    {el: 0 for el in new_file['Url']}
    dictionary = dict(zip(new_file['Url'], d.values()))

    dict2 = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    match_key = list(dict2.keys())[0]
    #match_value = list(dict2.values())[0]
    match2 = list(dict2.values())[1]

    return match_key, match2



st.image("magicien.png", width=100)
st.title("Redirects Wizard")

st.sidebar.header("Setup")

algorithm_selectbox = st.sidebar.radio(
    "1. Select the matching algorithm",
    ("Strings similarity", "Semantic matching", "BOW (inactive)"), key="1"
)

old_urls_upload = st.sidebar.file_uploader(label="2. Upload Your Old URLs", type=["csv"], help="CSV file with \";\" separator")

if old_urls_upload:
    old_file = pd.read_csv(old_urls_upload, encoding='utf-8', sep=";")
    old_file_colnames = old_file.columns.tolist()
    select_old_column = st.sidebar.radio("Which columns to match?", old_file_colnames, key="2")
    checkbox_whole_old_url = st.sidebar.checkbox("Use the whole URL to matching", help = "By default, we will match the last URL elements.")
    st.write(len(old_file), " old urls successfully added")
    st.table(old_file[:10].style.highlight_null(null_color='#E7E7E7'))
else:
    old_file = "old_file_no_added"

new_urls_upload = st.sidebar.file_uploader(label="3. Upload Your New URLs", type=["csv"], help="CSV file with \";\" separator")

if new_urls_upload:
    new_file = pd.read_csv(new_urls_upload, encoding='utf-8', sep=";")
    new_file_colnames = new_file.columns.tolist()
    select_new_column = st.sidebar.radio("Which columns to match?", new_file_colnames, key="3")
    if checkbox_whole_old_url:
        checkbox_whole_new_url = st.sidebar.checkbox("Use the whole URL", help = "By default, we will match the last URL elements.", value=True)
    else:
        checkbox_whole_new_url = st.sidebar.checkbox("Use the whole URL", help = "By default, we will match the last URL elements.", value=False)

    st.write(len(new_file), " new urls successfully added")
    st.table(new_file[:10].style.highlight_null(null_color='#E7E7E7'))


    @st.cache(allow_output_mutation=True)
    def load_linguistic_model(name="sentence-transformers/distiluse-base-multilingual-cased-v2"):
        """Instantiate a sentence-level DistilBERT model."""
        return SentenceTransformer(name)



### DISPLAY RESULT ###

    if new_urls_upload and old_urls_upload:
        match_button = st.button('Match!')
        if match_button:
            st.header('Result :')

            if algorithm_selectbox == "Strings similarity":
                alg_strings_similarity()

            if algorithm_selectbox == "Semantic matching":
                #load_model_button = st.button("Load the linguistic model")
                #if load_model_button:
                model = load_linguistic_model()
                st.success('The linguistic model is successfully loaded!')

                stqdm.pandas()
                old_file['New Temp Url'] = old_file[select_old_column].progress_map(lambda y: bert(y, model))
                old_file['New Url'] = old_file['New Temp Url'].apply(lambda x: x.split(',')[0])
                old_file['Similarity Score'] = old_file['New Temp Url'].apply(lambda x: x.split(',')[1]).astype('float').round(3)

                st.success('All items have been successfully matched!')
                redirects_plan = old_file[['Url', 'New Url', 'Similarity Score']][:50]

                st.dataframe(redirects_plan.style.background_gradient(cmap='flare', subset=['Similarity Score']))

                @st.cache
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')


                csv = convert_df(redirects_plan)

                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='redirects_plan.csv',
                    mime='text/csv',
                )

            else:
                st.write("This algorithm is not yet active.")
