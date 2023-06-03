### IMPORT LIBRARIES ###

import streamlit as st
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', None)
import seaborn as sns
cm = sns.light_palette("red", as_cmap=True)
st.set_page_config(layout="wide")
pd.options.display.float_format = '{:.2%}'.format
from stqdm import stqdm
import os

# STRINGS LIBRARIES
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# TRANSFORMERS LIBRARIES
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# OPENAI LIBRARIES
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

password = '0987'

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

    @st.cache_data
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
    embeddings2 = model.encode(new_file[select_new_column], convert_to_tensor=True)

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

password_input = st.sidebar.text_input("Enter a password :", type="password")

algorithm_selectbox = st.sidebar.radio(
    "1. Select the matching algorithm",
    ("Strings similarity", "Semantic matching (BERT)", "Semantic matching (GPT)", "BOW (inactive)"), key="1"
)

old_urls_upload = st.sidebar.file_uploader(
    "Chargez votre fichier au format excel', type='xlsx'")

if old_urls_upload:
    old_file = pd.read_excel(old_urls_upload)
    old_file_colnames = old_file.columns.tolist()
    select_old_column = st.sidebar.radio("Which columns to match?", old_file_colnames, key="2")
    checkbox_whole_old_url = st.sidebar.checkbox("Use the whole URL to matching", help = "By default, we will match the last URL elements.")
    st.write(len(old_file), " old urls successfully added")

    with st.expander("Your source URLs", expanded=False):
        st.dataframe(old_file[:10].style.highlight_null(null_color='#E7E7E7'))
else:
    old_file = "old_file_no_added"

new_urls_upload = st.sidebar.file_uploader(
    "Chargez votre fichier au format excel.', type='xlsx'")

if new_urls_upload:
    new_file = pd.read_excel(new_urls_upload)
    new_file_colnames = new_file.columns.tolist()
    select_new_column = st.sidebar.radio("Which columns to match?", new_file_colnames, key="3")
    if checkbox_whole_old_url:
        checkbox_whole_new_url = st.sidebar.checkbox("Use the whole URL", help = "By default, we will match the last URL elements.", value=True)
    else:
        checkbox_whole_new_url = st.sidebar.checkbox("Use the whole URL", help = "By default, we will match the last URL elements.", value=False)

    st.write(len(new_file), " new urls successfully added")

    with st.expander("Your source URLs", expanded=False):
        st.dataframe(new_file[:10].style.highlight_null(null_color='#E7E7E7'))


    @st.cache_resource
    def load_linguistic_model(name="sentence-transformers/distiluse-base-multilingual-cased-v2"):
    #def load_linguistic_model(name="sentence-transformers/all-MiniLM-L6-v2"):
        """Instantiate a sentence-level DistilBERT model."""
        return SentenceTransformer(name)


    def create_embedding(text):
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embeddings = response['data'][0]['embedding']
        return embeddings


### DISPLAY RESULT ###

    if new_urls_upload and old_urls_upload and password_input == '0987':
        match_button = st.button('Match!')
        if match_button:
            st.header('Result :')

            if algorithm_selectbox == "Strings similarity":
                alg_strings_similarity()

            if algorithm_selectbox == "Semantic matching (BERT)":
                #load_model_button = st.button("Load the linguistic model")
                #if load_model_button:

                model = load_linguistic_model()

                st.success('The linguistic model is successfully loaded!')

                stqdm.pandas()
                old_file['New Temp Url'] = old_file[select_old_column].progress_map(lambda y: bert(y, model))
                old_file['New Url'] = old_file['New Temp Url'].apply(lambda x: x.split(',')[0])
                old_file['Similarity Score'] = old_file['New Temp Url'].apply(lambda x: x.split(',')[1]).astype('float').round(3)

                st.success('All items have been successfully matched!')
                redirects_plan = old_file[['Url', 'New Url', 'Similarity Score']]

                st.table(redirects_plan[:50].style.background_gradient(cmap='flare', subset=['Similarity Score']))

                @st.cache_data
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

            if algorithm_selectbox == "Semantic matching (GPT)":
                st.success('Semantic matching (GPT)')

                os.environ['OPENAI_API_KEY'] = st.secrets['openai.api_key']
                
                stqdm.pandas()
                old_file['Embedding'] = old_file[select_old_column].progress_map(create_embedding)

                with st.expander("Embeddings pour le fichier source", expanded=False):
                    st.dataframe(old_file[:10])

                stqdm.pandas()
                new_file['Embedding'] = new_file[select_new_column].progress_map(create_embedding)

                with st.expander("Embeddings pour le fichier de destination", expanded=False):
                    st.dataframe(new_file[:10])


                def search_similar_pages(row, new_file):
                    # get the embedding of the current page in old_file
                    old_embedding = row['Embedding']

                    # compute the cosine similarity with all embeddings in new_file
                    new_file["similarity"] = new_file['Embedding'].apply(lambda x: cosine_similarity(x, old_embedding))

                    # find the row in new_file with maximum similarity
                    closest_match_row = new_file.loc[new_file['similarity'].idxmax()]

                    # extract closest page and its similarity
                    closest_page = closest_match_row['Url']
                    max_similarity = closest_match_row['similarity']

                    return pd.Series([closest_page, max_similarity])


                # Apply function to each row of old_file
                old_file[['Closest Page', 'Similarity']] = old_file.apply(lambda row: search_similar_pages(row, new_file),
                                                                          axis=1)

                st.dataframe(old_file[['Url', 'Closest Page', 'Similarity']].style.background_gradient(cmap='flare', subset=['Similarity']))




            else:
                st.write("This algorithm is not yet active.")
