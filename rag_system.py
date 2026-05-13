from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# function to preprocess uploaded pdf document
def chunk_embed(file_path):
    # get text from PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # split pdf into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(docs)

    # add chunks and their embeddings to vector database
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = Chroma(
        collection_name="pdf_chunks",
        embedding_function=embeddings
    )
    # Chroma creates embeddings for the chunks with the embedding function
    vector_store.add_documents(chunks)

    return vector_store

# function to return top 3 chunks (contains page content and metadata)
def run_query(question, vector_store):
    results = vector_store.similarity_search_with_score(question, k=3)
    top_chunks = []

    for chunk, _ in results:
        top_chunks.append(chunk)
    
    return top_chunks

# function to combine retrieved chunks into one string input for the LLM input
def combine_chunks(chunks):
    retrieved_chunks = []

    for chunk in chunks:
        retrieved_chunks.append(
            f"Page {chunk.metadata['page']}:\n"
            f"{chunk.page_content}"
        )
    combined_chunks = "\n\n".join(retrieved_chunks)

    return combined_chunks

# use LLM to answer query based on returned records
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Respond to the user's question based on information from the provided context.
        The context contains chunks that are separated by '\n\n'.
        Include a page number citation at the end of the response based on the chunk(s) used to create the answer.
        """
    ),
    ("user", "Question: {question}\nContext: {context}")
])

model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model

def generate_response(query, vector_store):
    context = run_query(query, vector_store)

    # generate response
    result = chain.invoke({"question": query, "context": context})
    response = result.content

    return response