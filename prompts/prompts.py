from langchain.prompts import PromptTemplate

# Define a template for generating responses based on the video context
video_based_answer_template = PromptTemplate(
    input_variables=["query", "context"], 
    template="""Answer the following question based only on the provided video content, dont answer from any external source:
    Context:   {context}
    Question: {query}

    Answer:"""
)