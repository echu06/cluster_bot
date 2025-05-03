# Python Coding Assistant using GPT-4-mini and Streamlit



import os
import io
import contextlib
import re
from dotenv import find_dotenv, load_dotenv
import streamlit as st
import pandas as pd# Python Coding Assistant using GPT-4-mini and Streamlit



import os
import io
import contextlib
import re
from dotenv import find_dotenv, load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import sklearn as sk
from sklearn.cluster import KMeans
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI # Look at the langchain and what libraries to select. 

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from openai import OpenAI


# Load environment variables from .env file
# os.chdir("/Users/ethanchu/Desktop/QAC 387/Project Folder")
load_dotenv()
client= OpenAI()
openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.load_local("vectorstore/faiss_index",embeddings,allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity",k=4)


from text_utilities import normalize_l2
from text_utilities import chunk_by_tokens_with_counts
from text_utilities import get_openai_embeddings
from text_utilities import save_embeddings
from text_utilities import clean_text
from text_utilities import find_optimal_clusters_elbow
from text_utilities import find_optimal_clusters_silhouette
from text_utilities import pca_graphing



# Streamlit page setup. 
@st.cache_data
def get_cached_embeddings(text_data):
    print("⏳ Generating embeddings... (not cached)")
    return get_openai_embeddings(text_data)


st.set_page_config(page_title="Python Data Analysis Assistant", layout="wide")
st.title("Python Data Analysis Assistant")
st.write(
    "Upload your data for text clustering! We will generate embeddings and create clusters for you!"
)

question = st.text_input("Tell us about the nature of the text data.", value="This is twitter data regarding political election opinions")

#Area for File Uploading:
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")


if uploaded_file:
    original_dataset = pd.read_csv(uploaded_file)
   
    df = original_dataset.copy()
    general_look = df.head().to_string(index=False) # This is a general look
    template = f"""
    Check this data given to you. {general_look} Do you think there is text data in there? 
    If no, only give one word "No". If yes, then give only the name of the column you think contains text. Only one column name in this format: 
    nameofcolumn
    Do not give any other information. 
    """

    # The f""" here is to insert the data head manually. 


    # Save this part as a separate file. 
    # Create a conditional argument that checks if the user already inputted some type of embeddings file. 
    # Use langchain to check for this condition and if so, then move onto the cluster analysis. 
    llm = ChatOpenAI(temperature=0)
    prompt = PromptTemplate(input_variables=[], template=template)
    llm_chain = prompt | llm 
    response = llm_chain.invoke({})
    flat_response = response.content
    if flat_response.strip().lower() == "no":
        st.write("Sorry you have no text data! Please try again next time!")

    else: # Clean clean clean!
        st.write("testing the response")
        st.write(flat_response)
        st.write(df[flat_response].head())

        # Initiating the cleaning process. 
        df["cleaned_text"]= clean_text(df[flat_response])
        df_filtered = df.dropna(subset=["cleaned_text"]).drop_duplicates(subset=["cleaned_text"]).copy()
        df_filtered["cleaned_text"] = df_filtered["cleaned_text"].astype(str)
        df_filtered = df_filtered.reset_index(drop=True)

        # Getting text data from filtered subset!
        text_data = df_filtered["cleaned_text"].tolist()

        reduced_embeddings, token_counts, skipped_texts = get_cached_embeddings(text_data)

        # STEP 4: Drop skipped rows (index-based and text-based)
        skipped_indices = {item["index"] for item in skipped_texts if item.get("index") is not None}
        skipped_text_set = {item["text"] for item in skipped_texts}


        df_used = df_filtered[~df_filtered.index.isin(skipped_indices)]
        df_used = df_used[~df_used["cleaned_text"].isin(skipped_text_set)].reset_index(drop=True)

        # ✅ STEP 4B: Force alignment if one silent failure occurred
        if df_used.shape[0] != reduced_embeddings.shape[0]:
            print(f"⚠️ Force-aligning: df_used = {df_used.shape[0]}, embeddings = {reduced_embeddings.shape[0]}")
            dropped_row = df_used.iloc[reduced_embeddings.shape[0]]
            print("🔍 Dropped row due to silent failure:\n", dropped_row["cleaned_text"][:200])
            df_used = df_used.iloc[:reduced_embeddings.shape[0]]
        
        embeddings_df = save_embeddings(df_used, reduced_embeddings, token_counts, filename="embeddings.csv")

try:
    if not embeddings_df.empty:
        st.write("You have valid Embeddings! Do you want us to engage in cluster analysis and cluster your data?")
        if st.button("Generate Analysis?"):
            st.write(embeddings_df.head())
            embeddings_sample = embeddings_df.head(50).to_string(index=False) # Give the AI a small snippet of the data. 
            template = """
                You are an expert data analysis agent who is about to engage in cluster analysis! you have this as your sample of embeddings dataframe with the columns and a few rows. 
                {embeddings_sample}
                Check if there are any text, string, or token columns with the embeddings sample! Please check carefully! 
                This would include columns that are also categorical variable columns!!!! Check based on the sample you were given!
                

                We have the embeddings in a data frame called `embeddings_df`. I want you to drop those text, string, or token columns 
                that you just identified from the columns. 

                Do not save any objects to do this task. (Do not instantiate an instance of some local variable or new global variable.)
                Note! This is a schematic! Replace the names as needed based on your embeddings_sample that you were given. 

                Also ensure that `embeddings_df` is still a pandas data frame at the end of the code and not a numpy array. 

                Only include the explanation and the code in the suggestions output.

                """
            
            prompt = PromptTemplate(input_variables=["embeddings_sample"], template=template)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.2, openai_api_key=openai_api_key)
            llm_chain = prompt | llm 
            result = llm_chain.invoke(
                    {"embeddings_sample": embeddings_sample}
                )
            
            st.markdown("### Suggested Analysis & Code")
            output_text = result.content if hasattr(result, "content") else result
            st.markdown(output_text if output_text else "*No response received from LLM*")
            st.session_state.generated_code = output_text

            # Code Execution
            if "generated_code" in st.session_state:
                result = st.session_state.generated_code

                # Extract Python code block from the result
                # Essentially what this does is it searches for the string starting with the backtick ''' and string ending with backticks '''. 
                code_match = re.search(r"```python(.*?)```", result, re.DOTALL)  # The (*?) captures all the things inside that are python? 
                code_to_run = code_match.group(1).strip() if code_match else result

                st.markdown("# System Response:")
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    try:
                        local_env = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns, "embeddings_df": embeddings_df}
                        exec(code_to_run, local_env)
                        embeddings_df = local_env.get("embeddings_df")
                        silhouette= find_optimal_clusters_silhouette(embeddings_df)
                        elbow = find_optimal_clusters_elbow(embeddings_df)
                        
                    except Exception as e:
                        st.error(f"Error running code: {e}")
                
                if "token_counts" not in embeddings_df.columns: # start our cluster analysis and identify clusters. 
                    template = """
                                You are an expert data analyst, Emmanuel Kaparakis, who is about to engage in cluster analysis. 
                                You are tasked with determining the optimal amount of clusters based on our estimates we have.

                                Here is the silhouette score we found {silhouette_score} and the elbow score we found {elbow_score}. 
                                At the very end of your comment, I want you to write `optimal_clusters = n` where n is the number of clusters
                                you think we should be using. Thank you. 

                                """
                    prompt = PromptTemplate(input_variables=["silhouette_score", "elbow_score"], template=template)
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=.7, openai_api_key=openai_api_key)
                    llm_chain = prompt | llm 


                    result = llm_chain.invoke(
                            {"silhouette_score": silhouette, "elbow_score": elbow}
                        )
                    
                    output_text = result.content if hasattr(result, "content") else result
                    st.write(output_text if output_text else "*No response received from LLM*")

                    if "optimal_clusters =" in output_text:
                        match = re.search(r"optimal_clusters\s*=\s*(\d+)", output_text)
                        optimal_clusters = int(match.group(1))
                        st.write(f"This is the optimal amount of clusters {optimal_clusters}")

                        if type(optimal_clusters) == int: # start your Cluster analysis:
                            df_filtered_aligned = df_filtered.loc[embeddings_df.index.intersection(df_filtered.index)].copy()
                            df_aligned = df.loc[embeddings_df.index.intersection(df.index)].copy()

                            # Then proceed with KMeans
                            kmeans = KMeans(n_clusters=optimal_clusters, init='random', n_init=10, random_state=0)
                            clusterlabels = kmeans.fit_predict(embeddings_df)

                            # Assign labels
                            df_filtered_aligned["clusterlabels"] = clusterlabels
                            df_aligned["clusterlabels"] = clusterlabels


                            cluster_summaries = {} # Create an empty list to store cluster summaries 
                            for cluster in sorted(df_aligned['clusterlabels'].unique()):
                                cluster_df = df_aligned[df_aligned['clusterlabels'] == cluster][flat_response].dropna().astype(str)
                                sample_size = min(200, len(cluster_df))
                                cluster_texts = " ".join(df_aligned[df_aligned['clusterlabels'] == cluster][flat_response].dropna().astype(str).sample(n=sample_size, random_state=42).tolist()) 
                                #prompt = f"""
                                #Please give me a recommendation about my clusters from the snippets that you see here. \n\n{cluster_texts}\n\nSummary:
                                #"""
                                docs = retriever.get_relevant_documents(question)
                                context_text = "\n\n".join(d.page_content for d in docs)

                                template = """
                                You are an expert on thematic analysis, named Michael Roth. Please make use of of the following information as you see fit, but you don't HAVE to:

                                {context_text}

                                Please help me summarize my cluster based on the following text snippets. Keep the summaries to a succinct 2 or 3 sentences. 
                                I want you to summarize all the text you are seeing, but do not summarize just by row by row. Summarize all the rows in the clusters together as one. Got it?

                                {cluster_texts}

                                """
                                prompt = PromptTemplate(
                                    input_variables=["context_text", "cluster_texts"],
                                    template=template
                                )
                                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, openai_api_key=openai_api_key)
                                llm_chain=prompt | llm
                                result = llm_chain.invoke({"context_text": context_text, "cluster_texts": cluster_texts})

                                #backup if this stuff above breaks.
                                #response = client.chat.completions.create(
                                #model="gpt-4o-mini",
                                #messages=[{"role": "user", "content": prompt}])

                                cluster_summaries[cluster] = result.content
                                combined_summary = ""
                            for cluster, summary in cluster_summaries.items():
                                 combined_summary += f"### Cluster {cluster} Summary\n{summary}\n\n"
                            st.success(f"""{combined_summary} """)

                else:
                    st.write("Sorry but system couldn't data manage properly! Sorry!")

except NameError:
    st.write("No embeddings: sorry!")
