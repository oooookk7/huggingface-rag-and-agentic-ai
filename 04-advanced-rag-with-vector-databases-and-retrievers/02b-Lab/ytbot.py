import os
import re
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi


_BASE_DIR = Path(__file__).resolve().parent
_ENV_CANDIDATES = [_BASE_DIR / ".env", _BASE_DIR.parent / ".env"]
for _env_path in _ENV_CANDIDATES:
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
        break
 
def get_video_id(url):    
    # Regex pattern to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None
 
def get_transcript(url):
    # Extracts the video ID from the URL
    video_id = get_video_id(url)
 
    if not video_id:
        return None

    # Fetch the list of available transcripts for the given YouTube video.
    # Support both newer and older youtube-transcript-api variants.
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
    except AttributeError:
        transcripts = YouTubeTranscriptApi().list(video_id)
   
    transcript = ""
    for t in transcripts:
        # Check if the transcript's language is English
        if t.language_code == 'en':
            if t.is_generated:
                # If no transcript has been set yet, use the auto-generated one
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                # If a manually created transcript is found, use it (overrides auto-generated)
                transcript = t.fetch()
                break  # Prioritize the manually created transcript, exit the loop
   
    return transcript if transcript else None
 
 
def process(transcript):
    # Initialize an empty string to hold the formatted transcript
    txt = ""
   
    # Loop through each entry in the transcript
    for i in transcript:
        try:
            # Append the text and its start time to the output string
            #txt += f"Text: {i['text']} Start: {i['start']}\n"
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            # If there is an issue accessing 'text' or 'start', skip this entry
            pass
           
    # Return the processed transcript as a single string
    return txt
 
def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
 
    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks
 
 
def setup_credentials():
    model_id = os.getenv("MINIMAX_MODEL_ID", "MiniMaxAI/MiniMax-M2.5")
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

    if not token:
        raise ValueError(
            "Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) in ../.env."
        )

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    return model_id, token

def define_parameters():
    # Return generation parameters for the MiniMax model hosted on Hugging Face
    return {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_new_tokens": 900,
    }
 
 
def initialize_minimax_llm(model_id, parameters):
    # Create and return a Hugging Face endpoint wrapped as a chat LLM
    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task="conversational",
        temperature=parameters["temperature"],
        top_p=parameters["top_p"],
        max_new_tokens=parameters["max_new_tokens"],
    )
    return ChatHuggingFace(llm=endpoint)
 
 
 
def setup_embedding_model():
    # Create and return a Hugging Face sentence-transformers embedding model
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
 
 
 
def create_faiss_index(chunks, embedding_model):
    """
    Create a FAISS index from text chunks using the specified embedding model.
   
    :param chunks: List of text chunks
    :param embedding_model: The embedding model to use
    :return: FAISS index
    """
    # Use the FAISS library to create an index from the provided text chunks
    return FAISS.from_texts(chunks, embedding_model)
 
 
 
def perform_similarity_search(faiss_index, query, k=3):
    """
    Search for specific queries within the embedded transcript using the FAISS index.
   
    :param faiss_index: The FAISS index containing embedded text chunks
    :param query: The text input for the similarity search
    :param k: The number of similar results to return (default is 3)
    :return: List of similar results
    """
    # Perform the similarity search using the FAISS index
    results = faiss_index.similarity_search(query, k=k)
    return results
 
 
def create_summary_prompt():
    """
    Create a PromptTemplate for summarizing a YouTube video transcript.
   
    :return: PromptTemplate object
    """
    # Define the template for the summary prompt
    template = """
    You are an AI assistant tasked with summarizing YouTube video transcripts.
    Provide a concise, informative summary that captures the main points.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore timestamps.
    3. Focus on spoken content only.

    Transcript:
    {transcript}

    Summary:
    """
   
    # Create the PromptTemplate object with the defined template
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )
   
    return prompt
 
 
def create_summary_chain(llm, prompt, verbose=True):
    """
    Create an LLMChain for generating summaries.
   
    :param llm: Language model instance
    :param prompt: PromptTemplate instance
    :param verbose: Boolean to enable verbose output (default: True)
    :return: LLMChain instance
    """
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)
 
 
def retrieve(query, faiss_index, k=7):
    """
    Retrieve relevant context from the FAISS index based on the user's query.
 
    Parameters:
        query (str): The user's query string.
        faiss_index (FAISS): The FAISS index containing the embedded documents.
        k (int, optional): The number of most relevant documents to retrieve (default is 3).
 
    Returns:
        list: A list of the k most relevant documents (or document chunks).
    """
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context
 
def create_qa_prompt_template():
    """
    Create a PromptTemplate for question answering based on video content.
    Returns:
        PromptTemplate: A PromptTemplate object configured for Q&A tasks.
    """
   
    # Define the template string
    qa_template = """
    You are an expert assistant answering questions using the provided video context.
    Requirements:
    1. Be accurate and avoid repetition.
    2. Stay grounded in the provided context.
    3. Keep the answer well-organized and direct.
    4. If context is insufficient, state that clearly.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    # Create the PromptTemplate object
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )
    return prompt_template
 
 
def create_qa_chain(llm, prompt_template, verbose=True):
    """
    Create an LLMChain for question answering.
 
    Args:
        llm: Language model instance
            The language model to use in the chain.
        prompt_template: PromptTemplate
            The prompt template to use for structuring inputs to the language model.
        verbose: bool, optional (default=True)
            Whether to enable verbose output for the chain.
 
    Returns:
        LLMChain: An instantiated LLMChain ready for question answering.
    """
   
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)
 
 
def generate_answer(question, faiss_index, qa_chain, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.
 
    Args:
        question: str
            The user's question.
        faiss_index: FAISS
            The FAISS index containing the embedded documents.
        qa_chain: LLMChain
            The question-answering chain (LLMChain) to use for generating answers.
        k: int, optional (default=3)
            The number of relevant documents to retrieve.
 
    Returns:
        str: The generated answer to the user's question.
    """
 
    # Retrieve relevant context
    relevant_context = retrieve(question, faiss_index, k=k)
 
    # Generate answer using the QA chain
    context_text = "\n\n".join(doc.page_content for doc in relevant_context)
    answer = qa_chain.predict(context=context_text, question=question)
 
    return answer
 
 
# Initialize an empty string to store the processed transcript after fetching and preprocessing
processed_transcript = ""
 
def summarize_video(video_url):
    """
    Title: Summarize Video
 
    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.
 
    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
 
    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    global fetched_transcript, processed_transcript
   
   
    if video_url:
        # Fetch and preprocess transcript
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."
 
    if processed_transcript:
        # Step 1: Set up Hugging Face MiniMax configuration
        model_id, _ = setup_credentials()
 
        # Step 2: Initialize MiniMax LLM for summarization
        llm = initialize_minimax_llm(model_id, define_parameters())
 
        # Step 3: Create the summary prompt and chain
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)
 
        # Step 4: Generate the video summary
        summary = summary_chain.run({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."
 
 
def answer_question(video_url, user_question):
    """
    Title: Answer User's Question
 
    Description:
    This function retrieves relevant context from the FAISS index based on the user’s query
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.
 
    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.
 
    Returns:
        str: The answer to the user's question or a message indicating that the transcript
             has not been fetched.
    """
    global fetched_transcript, processed_transcript
 
    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."
 
    if processed_transcript and user_question:
        # Step 1: Chunk the transcript (only for Q&A)
        chunks = chunk_transcript(processed_transcript)
 
        # Step 2: Set up Hugging Face MiniMax configuration
        model_id, _ = setup_credentials()
 
        # Step 3: Initialize MiniMax LLM for Q&A
        llm = initialize_minimax_llm(model_id, define_parameters())
 
        # Step 4: Create FAISS index for transcript chunks (only needed for Q&A)
        embedding_model = setup_embedding_model()
        faiss_index = create_faiss_index(chunks, embedding_model)
 
        # Step 5: Set up the Q&A prompt and chain
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)
 
        # Step 6: Generate the answer using FAISS index
        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."
 
 
 
with gr.Blocks() as interface:
 
    gr.Markdown(
        "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>"
    )
 
    # Input field for YouTube URL
    video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")
   
    # Outputs for summary and answer
    summary_output = gr.Textbox(label="Video Summary", lines=5)
    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to Your Question", lines=5)
 
    # Buttons for selecting functionalities after fetching transcript
    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")
 
    # Display status message for transcript fetch
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)
 
    # Set up button actions
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)
 
# Launch the app with configurable host/port.
if __name__ == "__main__":
    interface.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
