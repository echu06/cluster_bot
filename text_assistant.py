# Python Coding Assistant using GPT-4-mini and Streamlit



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
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI # Look at the langchain and what libraries to select. 


# Load environment variables from .env file
# os.chdir("/Users/ethanchu/Desktop/QAC 387/Project Folder")
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


from text_utilities import normalize_l2
from text_utilities import chunk_by_tokens_with_counts
from text_utilities import get_openai_embeddings
from text_utilities import save_embeddings
from text_utilities import clean_text
from text_utilities import find_optimal_clusters_elbow
from text_utilities import find_optimal_clusters_silhouette



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


#Area for File Uploading:
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")


if uploaded_file:
    df = pd.read_csv(uploaded_file)
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
        
        embeddings_df = save_embeddings(df_used, reduced_embeddings, token_counts, filename="huffpost_embeddings.csv")

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
                    
                else:
                    st.write("Sorry but system couldn't data manage properly! Sorry!")

    else:
        st.write("There are no embeddings!")

except NameError:
