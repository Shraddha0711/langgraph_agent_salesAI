import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END

# Initialize LLM and embeddings
llm_feedback = ChatOpenAI(model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()

# Load transcript and documents
def extract_text_from_docx(file_path):
    from docx import Document
    doc = Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return "\n".join(text)

# Define a GraphState for storing the context
from typing_extensions import TypedDict

class GraphState(TypedDict):
    transcript: str
    retrieved_knowledge: str
    feedback: str

# Load your .doc file into the text
transcript_file_path = "customer_service_transcript.docx"
transcript_text = extract_text_from_docx(transcript_file_path)

# Load and split documents into chunks
documents = [transcript_text]  # Add more documents as necessary
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create the FAISS index
vectorstore = FAISS.from_documents(texts, embeddings)

# Define the nodes for feedback generation

def generate_feedback(state):
    transcript = state["transcript"]
    
    # Create feedback prompt using the transcript
    feedback_prompt_text = f"""
    scenario_and_transcription : {transcript}
    Based on the given scenario and transcript, provide detailed feedback on the customer service agent’s performance. Analyze the call in the context of customer support and offer specific tips for how the agent can improve. Include what the agent did well, areas where they could improve, and actionable advice for future calls. Break the feedback into Concept and Application. Provide at least 4 tips for improvement and 2 things the agent did well.
    """
    
    # Retrieve relevant documents using FAISS index
    relevant_docs = vectorstore.similarity_search(transcript, k=5)
    retrieved_docs = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Combine the feedback prompt with retrieved documents
    combined_prompt = f"{feedback_prompt_text}\nRetrieved Knowledge: {retrieved_docs}"
    
    # Generate feedback using the LLM
    chain = ChatPromptTemplate.from_template(combined_prompt) | llm_feedback | StrOutputParser()
    feedback_result = chain.invoke({"transcript": transcript})
    
    state["feedback"] = feedback_result
    return state

# Define the graph
workflow = StateGraph(GraphState)

# Add the generate_feedback node
workflow.add_node("generate_feedback", generate_feedback)

# Define edges between nodes
workflow.add_edge("generate_feedback", END)

# Set the entrypoint
workflow.set_entry_point("generate_feedback")

# Compile the graph
graph_app = workflow.compile()

# Execute the workflow to generate feedback
state = {
    "transcript": transcript_text,
    "retrieved_knowledge": "",  # Knowledge will be retrieved dynamically
    "feedback": ""
}
feedback_result = graph_app.invoke(state)
print(feedback_result["feedback"])
