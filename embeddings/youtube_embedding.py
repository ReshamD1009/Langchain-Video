from langchain_community.vectorstores import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import PGVector
import os
from dotenv import load_dotenv
load_dotenv()

DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings_generator = GoogleGenerativeAIEmbeddings(
            api_key=GOOGLE_API_KEY,
            model="models/embedding-001"  
        )

def store_embeddings_pgvector(texts):
    try:
        

        if not DB_CONNECTION_URL_2:
            raise ValueError("Database connection URL is missing!")

        vectorstore = PGVector(
            connection_string=DB_CONNECTION_URL_2,
            embedding_function=embeddings_generator,
        )

        vectorstore.add_texts(texts)
        print(f"Data successfully stored in the  table.")
    except Exception as e:
        print(f"Error storing embeddings in PGVector: {e}")


def similarity_search_with_response(query):
    try:
        
        embeddings_generator = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY, model="models/embedding-001")
        vectorstore = PGVector(
            connection_string=os.getenv("DB_CONNECTION_URL_2"),
            embedding_function=embeddings_generator
        )
        
        search_results = vectorstore.similarity_search(query, k=3)  

        if not search_results:
            return "Sorry, I couldn't find relevant information from the video content."

        # Concatenate the top search results for generating the response
        context = "\n".join([result.page_content for result in search_results])
        
        prompt_template = PromptTemplate(
            input_variables=["query", "context"], 
            template="""
            Answer the following question based only on the provided video content:
            Context:{context}
            Question: {query}
            Answer:
            """
        )
        
        chatgroq = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-8b-8192",  
            temperature=0.0,
            max_retries=2
        )

        llm_chain = prompt_template | chatgroq

        response = llm_chain.invoke({"query": query, "context": context})

        return response

    except Exception as e:
        print(f"Error in similarity search or LLM response generation: {e}")
        return "An error occurred while processing your query."