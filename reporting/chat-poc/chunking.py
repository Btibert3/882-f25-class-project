import streamlit as st
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core import Document

st.image("https://questromworld.bu.edu/ftmba/wp-content/uploads/sites/42/2021/11/Questrom-1-1.png")


st.title("LlamaIndex")
st.markdown("https://docs.llamaindex.ai/en/stable/")

# Sidebar: Chunking Options
st.sidebar.header("Chunking Options")
chunk_strategy = st.sidebar.selectbox(
    "Choose a Chunking Strategy",
    ["Fixed Size", "Semantic (Sentences)", "Paragraph-based", "RecursiveCharacterTextSplitter"]
)
chunk_size = st.sidebar.slider("Chunk Size (tokens/words)", 0, 1000, step=10, value=300)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 100, step=2, value=50)

st.sidebar.markdown("---")

st.markdown("### Sample Text (optional)")

sample_text = """
Data teams are increasingly being asked to upskill with generative AI because the nature of analytics pipelines is changing faster than most organizations can keep up with. Traditional ETL workflows were designed for structured data and predictable transformations, but modern pipelines must ingest unstructured text, audio, images, logs, and documents that require intelligence—not just automation.

Generative AI models are now being embedded at every stage of the data lifecycle. In ingestion, LLMs help classify messy data, detect anomalies, and auto-tag content that historically required manual review. During transformation, they assist with feature engineering by generating embeddings, extracting entities, summarizing large documents, and normalizing inconsistent fields. Even in downstream consumption layers, generative agents can write SQL, explain business metrics, and produce narrative insights.

Because of these shifts, companies expect data engineers, analysts, and analytics engineers to understand how to call LLM APIs, manage prompt templates, tune retrieval pipelines, and integrate model outputs into dbt, Airflow, or Prefect workflows. This is no longer “AI as a separate discipline”—it's becoming a standard skill in the analytics stack.

Another driver is that business stakeholders increasingly want faster iteration. Instead of waiting weeks for new dashboards or transformations, they expect conversational interfaces, automated documentation, semantic search, and AI-assisted analytics. Data teams sit closest to the raw data and are best positioned to operationalize these workflows—if they know how.

However, the adoption of generative AI brings new responsibilities. Teams must learn how to evaluate LLM outputs, manage hallucinations, govern data used in prompts, and ensure cost controls when running large-scale inference in production. Organizations that fail to invest in upskilling risk creating brittle pipelines or misusing the technology altogether.

As a result, upskilling is not optional. It is now a strategic requirement for data teams that want to stay relevant and build robust, intelligent analytics pipelines that leverage both deterministic code and generative models working together.
"""

with st.expander("Show sample text", expanded=False):
    st.markdown(sample_text)

# Text Input
st.header("Input Text")
input_text = st.text_area("Paste your text here:", height=200)

# Process Text
if st.button("Chunk Text"):
    st.subheader("Chunked Output")

    if not input_text.strip():
        st.error("Please provide some text to chunk.")
    else:
        # Initialize the parser based on the selected strategy
        if chunk_strategy == "Fixed Size":
            parser = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = parser.split_text(input_text)
            st.sidebar.markdown("The TokenTextSplitter attempts to split to a consistent chunk size according to raw token counts.")
        elif chunk_strategy == "Semantic (Sentences)":
            parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = parser.split_text(input_text)
            st.sidebar.markdown("The SentenceSplitter attempts to split text while respecting the boundaries of sentences.")
        elif chunk_strategy == "Paragraph-based":
            # Paragraph-based chunking (splits by newlines)
            chunks = input_text.split("\n\n")  # Splitting by paragraphs
            st.sidebar.markdown("The settings above are ignored, looks for '`\n\n`' to define paragraphs.")
        elif chunk_strategy == "RecursiveCharacterTextSplitter":
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            parser = LangchainNodeParser(text_splitter)
            document = Document(
                text=input_text,  # LlamaIndex uses 'text' instead of 'page_content'
                id_="doc-1"      # LlamaIndex uses 'id_' instead of 'id'
            )
            chunks = parser.get_nodes_from_documents([document])
            st.sidebar.markdown("""Llama-index plays nicely with Langchain! This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.""")

        # Display chunked output
        for idx, chunk in enumerate(chunks):
            st.write(f"**Chunk {idx+1}:**\n{chunk}\n")