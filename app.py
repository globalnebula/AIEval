import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, util
import PyPDF2

# Initialize Groq API client and Sentence Transformer model
client = Groq(api_key="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")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to load PDF and split into chunks
def load_pdf_and_chunk(pdf_file, chunk_size=10000):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Split the text into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Function to retrieve the most relevant chunk based on the question
def retrieve_relevant_chunk(question, chunks):
    embeddings_question = model.encode(question, convert_to_tensor=True)
    chunk_scores = []
    
    for chunk in chunks:
        embeddings_chunk = model.encode(chunk, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings_question, embeddings_chunk).item()
        chunk_scores.append((similarity, chunk))
    
    # Sort chunks by similarity and return the most relevant one
    most_relevant_chunk = sorted(chunk_scores, key=lambda x: x[0], reverse=True)[0][1]
    return most_relevant_chunk

# Prompt-tuned interaction with LLM
def interact_with_llm(question, student_answers, client, reference_text):
    completion = client.chat.completions.create(
        model="llama3-groq-70b-8192-tool-use-preview",
        messages=[
            {"role": "system", "content": """
                You are a strict teacher grading short-answer questions.
                Consider the following reference material and assign a score (out of 10) to the student answers based on:
                - Relevance to the question
                - Completeness of the answer
                - Accuracy of the information provided.
                """.strip()},
            {"role": "user", "content": f"Here is some reference material: {reference_text}. {question}"}
        ],
        temperature=0.5,  # Lower temperature for consistency
        max_tokens=1024,
        top_p=0.9,
        stream=False
    )
    
    # Access the generated ideal answer properly
    ideal_answer = completion.choices[0].message.content

    # Evaluate student answers
    results = []
    for student_answer in student_answers:
        score = evaluate_answer_similarity(ideal_answer, student_answer)
        results.append(score)

    return results

# Embedding-based similarity scoring for additional accuracy
def evaluate_answer_similarity(ideal_answer, student_answer):
    embeddings_ideal = model.encode(ideal_answer, convert_to_tensor=True)
    embeddings_student = model.encode(student_answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings_ideal, embeddings_student).item()

    score = round(similarity * 10, 2)  # Convert to a score out of 10
    return score

# Streamlit UI
st.title("Chunked Textbook RAG Evaluation System")

# File uploader for the textbook PDF
uploaded_pdf = st.file_uploader("Upload the textbook (PDF)", type=["pdf"])

if uploaded_pdf is not None:
    # Load and chunk the textbook into manageable parts
    chunks = load_pdf_and_chunk(uploaded_pdf)
    st.success("Textbook uploaded and chunked successfully!")

    # Input for the question and student answers
    question = st.text_input("Enter the question:")
    student_answers_input = st.text_area("Enter comma-separated student answers:")

    if st.button("Evaluate"):
        if question and student_answers_input:
            # Process student answers as a list
            student_answers = [answer.strip() for answer in student_answers_input.split(",")]

            # Retrieve the most relevant chunk for the question
            reference_text = retrieve_relevant_chunk(question, chunks)

            # Interact with the LLM and get scores for each student answer
            scores = interact_with_llm(question, student_answers, client, reference_text)

            # Display the scores for each student
            for i, score in enumerate(scores):
                st.write(f"Student {i + 1}'s score: {score}/10")
        else:
            st.write("Please provide both a question and student answers.")
