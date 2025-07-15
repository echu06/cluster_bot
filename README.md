# text_analysis

This directory contains code that will create an online application to automatically cluster text data. 

Please ensure that before downloading the text_assistant.py file, that you download the requirements.txt file, the text_utilities.py file, ragpipeline file, and thematic analysis pdf. The text_utilities file contains all of the functions necessary for the assistant to create embeddings, ragpipeline file is how one would implement rag, and the thematic analysis pdf is used for the AI to better respond to our application. This program was mainly made to cluster data and give summaries on the clusters shown. 

We've managed to do the cluster analysis and use the file thematic analysis to feed into the AI and improve the analysis that the AI can give. Additionallly, we decided to ask for context in the beginning of the streamlit app where the person who is uploading the dataset can tell the AI what the dataset is about. This will give a new piece of context to the AI about what the patterns they should be seeing would look like. 

It should also be noted that this code might not run with RAG if you are using an M1 macbook, that of which I encountered. 

