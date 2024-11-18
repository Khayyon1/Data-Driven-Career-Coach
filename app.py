import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import pandas as pd
import matplotlib.pyplot as plt
import pdfplumber

# Set OpenAI API key from Streamlit secrets
api_key = st.secrets['openai']['api_key']

# Create the LLM
llm = OpenAI(openai_api_key=api_key, temperature=0)

def process_file(file, file_type='pdf'):
    if file_type == 'pdf':
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file_type == 'txt':
        return file.read().decode("utf-8")
    return ""

def analyze_resume_vs_job(resume_text, job_desc_text):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    resume_docsearch = FAISS.from_texts([resume_text], embeddings)
    job_desc_docsearch = FAISS.from_texts([job_desc_text], embeddings)

    # Key Matches
    results = resume_docsearch.similarity_search(job_desc_text, k=5)
    matches = [match.page_content for match in results]

    #Skills gap Analysis
    prompt = PromptTemplate(input_variables=["resume", "job_desc"], template="""
    Analyze the following resume and job description. Highlight the missing skills in the resume
    that are critical for the job description and suggest improvements.
    Resume:
    {resume}

    Job Description:
    {job_desc}                                                     
""",)
    
    chain = LLMChain(llm=llm, prompt=prompt)
    analysis = chain.run({"resume": resume_text, "job_desc": job_desc_text})

    return matches, analysis

def plot_skill_recommendations():
    skills_data = {
        "Skills": ["Python", "AWS", "NLP", "Streamlit", "Leadership"],
        "Relevance": [90, 80, 75, 70, 65]
    }

    df_skills = pd.DataFrame(skills_data)
    fig, ax = plt.subplots()
    ax.barh(df_skills['Skills'], df_skills['Relevance'], color='skyblue')
    ax.set_xlabel("Relevance (%)")
    ax.set_title("Skill Recommendations")
    st.pyplot(fig)

def generate_mock_questions(job_desc_text): 
    interview_prompt = PromptTemplate(input_variables=["job_desc"], template="""
    Based on the following job description, generate 5 relevant interview questions: {job_desc}
""")
    interview_chain = LLMChain(llm=llm, prompt=interview_prompt)
    questions = interview_chain.run({"job_desc": job_desc_text})
    return questions

def main():
    st.title(":chart Data-Driven Career Mentor")
    st.write("Upload your resume and job_description to get tailored career advice")

    # File Uploads
    resume_file = st.file_uploader("Upload your Resume(PDF)", type=["pdf"])
    job_desc_file = st.file_uploader("Upload Job Description (PDF or TXT)", type=["pdf", "txt"])

    if resume_file and job_desc_file:
        resume_text = process_file(resume_file, 'pdf')
        job_desc_text = process_file(job_desc_file, 'pdf' if job_desc_file.name.endswith('.pdf') else "txt" )

        # Analyze Resume and Job Description
        matches, analysis = analyze_resume_vs_job(resume_text, job_desc_text)

        st.subheader("Key Matches Between Resume and Job Description")
        for i, match in enumerate(matches):
            st.write(f"**Match {i+1}:** {match}")
            st.subheader("Skill Gap Analysis")
            st.write(analysis)

            #skill rec visuals
            st.subheader("Recommended Skills for Improvement")
            plot_skill_recommendations()

            st.subheader("Mock Interview Questions")
            if st.button("Generate Questions"):
                questions = generate_mock_questions(job_desc_text)
                st.write(questions)
            else:
                st.info("Please Upload Both your resume and a job description to proceed")

if __name__ == "__main__":
    main()