import os, sys

from haystack.pipelines import Pipeline, GenerativeQAPipeline
from haystack.nodes import Crawler, PreProcessor, BM25Retriever, OpenAIAnswerGenerator, JsonConverter, TextConverter
from haystack.nodes import JoinDocuments, PromptNode, PromptTemplate
from haystack.nodes.ranker import DiversityRanker, SentenceTransformersRanker
from haystack.document_stores import InMemoryDocumentStore

import streamlit as st
import streamlit.components.v1 as components


MY_API_KEY = os.environ["OPENAI_API_KEY"]

@st.cache_resource
def create_index():
    document_store = InMemoryDocumentStore(use_bm25=True)

    doc_dir = "svpg"
    files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
    json_converter = JsonConverter()

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=270,
        split_respect_sentence_boundary=True,
    )

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_node(component=json_converter, name="JSONConverter", inputs=["File"])
    indexing_pipeline.add_node(component=preprocessor, name="preprocessor", inputs=['JSONConverter'])
    indexing_pipeline.add_node(component=document_store, name="document_store", inputs=['preprocessor'])

    indexing_pipeline.run(file_paths=files_to_index)
    return document_store

@st.cache_resource
def create_nodes():
    document_store = create_index()
    retriever = BM25Retriever(document_store=document_store)
    retriever.debug = True

    marty_prompt = PromptTemplate(prompt="""You are the Silicon Valley Product Group (SVPG), a group of product experts and authors of articles and multiple books on product management.
You consult and write about topics and answer questions on product management, product development and product organizations.
For that, you use the provided documents.
The documents are parts of your blogs and articles that are related to the question.
Write in your style similar to the documents.
Be concise and only use the information from the documents.
If you use information from a document then cite the source in form of [number of document].
For example, use [x] to indicate that you are using information from [x].
The citation must only use the number that is in brackets [x].
\n\nHere are the documents: {join(documents, delimiter=new_line, pattern=new_line+'document[$idx]: $content')}
\n\nQuestion: {query} \n\nSVPG:""")

    prompt_node = PromptNode(model_name_or_path="gpt-3.5-turbo", 
                        default_prompt_template=marty_prompt,
                        api_key=MY_API_KEY,
                        max_length=300,
                        model_kwargs={"temperature":.1, "frequency_penalty":0.7}
                        )
    divranker = DiversityRanker(
		                model_name_or_path="all-MiniLM-L6-v2",
  		                top_k=3,
  				        use_gpu=False,
  				        similarity="dot_product"
	                    )
    sentranker = SentenceTransformersRanker(
                        model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2",
                        top_k=5,
                        use_gpu=False,
                        batch_size=20
                        )
    return retriever, prompt_node, divranker, sentranker

# here the streamlit app starts
st.set_page_config(layout="wide", page_icon="🤖")
st.title("ChatSVPG")

# get user input
question = st.text_input("Your question to Silicon Valley Product Group:")

# prepare pipeline
retriever, prompt_node, divranker, sentranker = create_nodes()
p = Pipeline()
p.add_node(component=retriever, name="Retriever", inputs=["Query"])
p.add_node(component=sentranker, name="SentRanker", inputs=["Retriever"])
p.add_node(component=divranker, name="DivRanker", inputs=["SentRanker"])
p.add_node(component=prompt_node, name="QAPromptNode", inputs=["DivRanker"])
 
# send query
if st.button("Ask!"):
    results = p.run(query=question, params={"Retriever": {"top_k": 10}})
   
    answer = results["results"][0]
    docs = results["invocation_context"]["documents"]
    prompt = results["invocation_context"]["prompts"][0]
    urls = [(doc.meta["url"], doc.meta["_split_id"]) for doc in docs]
    print(prompt, file=sys.stderr)
    print(answer, file=sys.stderr)
    print(docs, file=sys.stderr)

    with st.container():
        st.markdown(f"> {answer}")
        for url, sid in urls:
            st.markdown(f"""[{url}]({url}) {sid}\n""")
