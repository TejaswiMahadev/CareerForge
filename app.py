import streamlit as st
import json
import requests
from datetime import datetime
import xml.etree.ElementTree as ET 
import time 
import re
import tempfile
import os
from dotenv import load_dotenv
import google.generativeai as genai
import PyPDF2
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERP_API_KEY = os.getenv("SERPAPI_KEY")  

genai.configure(api_key=GOOGLE_API_KEY) 

# Set page configuration
st.set_page_config(page_title="CareerForge", page_icon="🚀", layout="wide")

# Title and description
st.title("Career Forge AI ")
st.markdown("Plan your professional journey with AI-powered career insights")

# Sidebar info (removing manual API input)
with st.sidebar:
    st.header("Configuration")
    if GOOGLE_API_KEY:
        st.success("API Key Loaded from .env ✅")
    else:
        st.error("Google API Key not found. Please check your .env file.")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app extracts structured information from resumes using Google's Generative AI and Retrieval Augmented Generation (RAG).")

# Ensure API key is present before proceeding
if not GOOGLE_API_KEY:
    st.stop()  # Stop execution if API key is missing

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file) -> str:
    """Extracts raw text from a PDF file."""
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to create RAG pipeline
def create_rag_pipeline(text: str, api_key: str):
    """Creates a RAG-based pipeline to extract structured data using Google Gemini."""
    # Create a temporary file to store the text
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as temp_file:
            temp_file.write(text)
            temp_path = temp_file.name
        
        # Initialize models with Google API
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1,
            top_k=40,
            top_p=0.95,
        )
        
        # Split text into chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        
        # Load and split documents
        loader = TextLoader(temp_path, encoding="utf-8")  # Explicitly set encoding
        documents = loader.load()
        split_docs = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )
        
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except Exception as e:
            st.warning(f"Could not delete temporary file: {str(e)}")
        
        return qa_chain
    except Exception as e:
        st.error(f"Error in RAG pipeline: {str(e)}")
        raise

# Function to parse resume
def parse_resume(pdf_file, api_key: str) -> dict:
    """Parses a resume PDF and extracts structured information using RAG with Google Gemini."""
    raw_text = extract_text_from_pdf(pdf_file)
    qa_chain = create_rag_pipeline(raw_text, api_key)
    
    # Define queries for extracting structured information
    queries = {
        "contact_info": "What is the contact information of this candidate including name, email, phone number, and location?",
        "skills": "What technical and soft skills does this candidate have? List them in detail.",
        "education": "What is the educational background of this candidate? Include institution names, degrees, years, and any notable achievements.",
        "experience": "What work experience does this candidate have? Include company names, job titles, dates, and key responsibilities or achievements.",
        "projects": "What projects has this candidate worked on? Include project names, technologies used, and outcomes.",
        "certifications": "What certifications does this candidate have?",
        "languages": "What languages does this candidate speak or know?",
        "summary": "Write a 2-3 sentence professional summary of this candidate based on their resume."
    }
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    structured_data = {}
    total_queries = len(queries)
    
    # Process each query and update progress
    for i, (key, query) in enumerate(queries.items()):
        progress_text.text(f"Extracting {key}...")
        try:
            result = qa_chain.run(query)
            structured_data[key] = result
        except Exception as e:
            structured_data[key] = f"Error extracting this information: {str(e)}"
        
        # Update progress
        progress = (i + 1) / total_queries
        progress_bar.progress(progress)
    
    progress_text.text("Extraction complete!")
    return structured_data

# Main app flow
uploaded_file = st.file_uploader("Upload a Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    if not GOOGLE_API_KEY:
        st.warning("Please enter your Google API Key in the sidebar to proceed.")
    else:
        st.success("Resume uploaded successfully!")
        
        with st.spinner("Parsing resume... This may take a minute."):
            try:
                parsed_data = parse_resume(uploaded_file, GOOGLE_API_KEY)
                st.session_state.parsed_resume = parsed_data
                
                # Display the extracted information
                st.header("Extracted Information")
                
                # Contact Info
                if "contact_info" in parsed_data:
                    st.subheader("Contact Information")
                    st.write(parsed_data["contact_info"])
                
                # Summary
                if "summary" in parsed_data:
                    st.subheader("Professional Summary")
                    st.write(parsed_data["summary"])
                
                # Create columns for main sections
                col1, col2 = st.columns(2)
                
                with col1:
                    # Skills
                    if "skills" in parsed_data:
                        st.subheader("Skills")
                        st.write(parsed_data["skills"])
                    
                    # Education
                    if "education" in parsed_data:
                        st.subheader("Education")
                        st.write(parsed_data["education"])
                    
                    # Languages
                    if "languages" in parsed_data:
                        st.subheader("Languages")
                        st.write(parsed_data["languages"])
                
                with col2:
                    # Experience
                    if "experience" in parsed_data:
                        st.subheader("Experience")
                        st.write(parsed_data["experience"])
                    
                    # Projects
                    if "projects" in parsed_data:
                        st.subheader("Projects")
                        st.write(parsed_data["projects"])
                    
                    # Certifications
                    if "certifications" in parsed_data:
                        st.subheader("Certifications")
                        st.write(parsed_data["certifications"])
                
                # JSON Output
                st.header("JSON Output")
                st.json(parsed_data)
                
                # Download button for JSON
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(parsed_data, indent=4),
                    file_name="resume_parsed.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"An error occurred while parsing the resume: {str(e)}")
                st.error("Make sure your Google API Key is valid and has access to Gemini models.")

# ------------------------------------------------ RESUME PARSING -----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
CAREER_TEMPLATES = {
    "data scientist": {
        "starting_point": "Data enthusiast with basic programming knowledge",
        "end_goal": "Principal Data Scientist or AI/ML Director",
        "progression_levels": [
            {
                "level": "Level 1: Foundation Building",
                "skills_to_acquire": ["Basic Python/R", "Statistics fundamentals", "SQL basics", "Data visualization basics"],
                "learning_resources": ["Online courses in Python/R", "Khan Academy statistics", "SQL tutorials"],
                "projects": ["Analyze public datasets", "Create basic visualizations", "Build a data cleaning pipeline"],
                "time_estimate": "3-6 months"
            },
            {
                "level": "Level 2: Junior Data Analyst",
                "skills_to_acquire": ["Advanced SQL", "Data wrangling", "Business analytics", "Dashboard creation"],
                "learning_resources": ["DataCamp courses", "Business analytics books", "Tableau/PowerBI tutorials"],
                "projects": ["Create interactive dashboards", "Perform A/B testing", "Build automated reports"],
                "time_estimate": "6-12 months"
            },
            {
                "level": "Level 3: Data Scientist Beginner",
                "skills_to_acquire": ["Machine learning basics", "Feature engineering", "Model evaluation", "Statistical analysis"],
                "learning_resources": ["Andrew Ng's ML course", "Kaggle competitions", "Applied ML books"],
                "projects": ["Predictive models", "Classification systems", "Recommendation engines"],
                "time_estimate": "12-18 months"
            },
            {
                "level": "Level 4: Mid-level Data Scientist",
                "skills_to_acquire": ["Advanced ML algorithms", "Deep learning", "MLOps basics", "Experiment design"],
                "learning_resources": ["Deep learning specializations", "MLOps courses", "Industry conferences"],
                "projects": ["End-to-end ML systems", "Neural network implementations", "ML pipeline creation"],
                "time_estimate": "18-24 months"
            },
            {
                "level": "Level 5: Senior Data Scientist",
                "skills_to_acquire": ["ML system architecture", "Team leadership", "Business strategy", "Advanced specialized techniques"],
                "learning_resources": ["Leadership courses", "Advanced specialized ML books", "Industry conferences"],
                "projects": ["Design data strategies", "Lead cross-functional projects", "Mentor junior scientists"],
                "time_estimate": "2-3 years"
            },
            {
                "level": "Level 6: Principal/Director",
                "skills_to_acquire": ["AI strategy development", "Executive communication", "Research direction", "Business innovation"],
                "learning_resources": ["Executive education", "Industry leadership forums", "Research journals"],
                "projects": ["Define organization-wide AI strategy", "Drive business transformation", "Research innovation"],
                "time_estimate": "3-5+ years"
            }
        ]
    },
    "software engineer": {
        "starting_point": "Coding enthusiast with basic programming knowledge",
        "end_goal": "Software Architect or CTO",
        "progression_levels": [
            {
                "level": "Level 1: Programming Foundations",
                "skills_to_acquire": ["Core programming language", "Basic data structures", "Version control (Git)", "Basic algorithms"],
                "learning_resources": ["Codecademy courses", "FreeCodeCamp", "Programming fundamentals books"],
                "projects": ["Simple CLI applications", "Basic web pages", "Small utility scripts"],
                "time_estimate": "3-6 months"
            },
            {
                "level": "Level 2: Junior Developer",
                "skills_to_acquire": ["Web development frameworks", "Databases", "API integration", "Testing basics"],
                "learning_resources": ["Framework documentation", "Database courses", "Testing tutorials"],
                "projects": ["Personal portfolio website", "CRUD applications", "Simple web apps"],
                "time_estimate": "6-12 months"
            },
            {
                "level": "Level 3: Software Engineer",
                "skills_to_acquire": ["System design basics", "Advanced frameworks", "Deployment processes", "Code optimization"],
                "learning_resources": ["System design primers", "Advanced framework courses", "DevOps tutorials"],
                "projects": ["Full-stack applications", "Microservices", "Continuous integration setups"],
                "time_estimate": "12-24 months"
            },
            {
                "level": "Level 4: Senior Software Engineer",
                "skills_to_acquire": ["Software architecture patterns", "Team leadership", "Technical decision making", "Performance tuning"],
                "learning_resources": ["Architecture books", "Leadership courses", "Advanced engineering blogs"],
                "projects": ["Design system architecture", "Lead development teams", "Implement complex systems"],
                "time_estimate": "2-4 years"
            },
            {
                "level": "Level 5: Technical Lead / Architect",
                "skills_to_acquire": ["Enterprise architecture", "Cross-team coordination", "Technical strategy", "Mentorship"],
                "learning_resources": ["Enterprise architecture courses", "Technical leadership books", "Industry conferences"],
                "projects": ["Design enterprise systems", "Technical roadmap creation", "Cross-functional leadership"],
                "time_estimate": "3-5 years"
            },
            {
                "level": "Level 6: CTO / VP Engineering",
                "skills_to_acquire": ["Technology vision", "Executive leadership", "Business strategy", "Innovation management"],
                "learning_resources": ["Executive education", "Business strategy courses", "Innovation management"],
                "projects": ["Technology vision and strategy", "Organization-wide initiatives", "Digital transformation"],
                "time_estimate": "5+ years"
            }
        ]
    }
}

# Function to generate career roadmap using Gemini
# def generate_gemini_roadmap(career, details=""):
#     try:
#         # Initialize Gemini model
#         model = genai.GenerativeModel('gemini-1.5-flash')

        
#         # Craft a detailed prompt for level-wise career roadmap
#         prompt = f"""
#         Create a detailed level-wise career roadmap for someone pursuing a career as a {career}.
#         {details if details else ''}
        
#         The roadmap should show clear progression from a starting point (beginner) to an end goal (expert/leader). 
#         Format the roadmap as a step-by-step guide with distinct levels of progression.
        
#         For each level in the progression, include:
        
#         # Career Roadmap: {career}
        
#         ## Starting Point
#         [Describe the minimum starting point - what should someone already know or have]
        
#         ## End Goal
#         [Describe the ultimate career position or expertise level this roadmap leads to]
        
#         ## Level-by-Level Progression
        
#         ### Level 1: [Name this level]
#         - **Skills to Acquire**: [List specific skills to learn at this level]
#         - **Learning Resources**: [Recommend specific resources]
#         - **Projects to Build**: [Suggest practical projects to demonstrate skills]
#         - **Estimated Time Investment**: [Provide realistic timeframe]
#         - **Transition Signal**: [How to know when ready to advance to next level]
        
#         ### Level 2: [Name this level]
#         [Same structure as Level 1]
        
#         [Continue with Levels 3-6 as needed]
        
#         ## Alternative Pathways
#         [Describe 2-3 alternative specializations or paths someone might take]
        
#         ## Common Obstacles and Solutions
#         [Describe 3-4 common challenges at different stages and how to overcome them]
        
#         Keep the information practical, actionable, and focused on concrete steps.
#         """
        
#         # Generate response from Gemini
#         response = model.generate_content(prompt)
        
#         # Return the formatted text
#         return response.text
    
#     except Exception as e:
#         st.error(f"Error generating roadmap with Gemini: {str(e)}")
#         # Fallback to template-based method
#         return generate_level_wise_roadmap(career)
def generate_gemini_roadmap(career, details="", resume_data=None):
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare resume context if resume_data is provided
        resume_context = ""
        if resume_data:
            resume_context = """
            Personalize the roadmap using the following resume data for the user:
            - **Contact Information**: {contact_info}
            - **Skills**: {skills}
            - **Education**: {education}
            - **Experience**: {experience}
            - **Projects**: {projects}
            - **Certifications**: {certifications}
            - **Languages**: {languages}
            - **Professional Summary**: {summary}
            
            Use this data to:
            - Set the starting point based on the user's current skills, education, and experience.
            - Tailor skills to acquire by identifying gaps between their current abilities and industry requirements for {career}.
            - Recommend resources and projects that build on their existing projects and certifications.
            - Adjust the end goal and progression levels to align with their professional summary and career trajectory.
            If any section is missing or marked as 'Error extracting this information', assume minimal knowledge in that area and adjust accordingly.
            """.format(
                contact_info=resume_data.get('contact_info', 'Not provided'),
                skills=resume_data.get('skills', 'Not provided'),
                education=resume_data.get('education', 'Not provided'),
                experience=resume_data.get('experience', 'Not provided'),
                projects=resume_data.get('projects', 'Not provided'),
                certifications=resume_data.get('certifications', 'Not provided'),
                languages=resume_data.get('languages', 'Not provided'),
                summary=resume_data.get('summary', 'Not provided'),
                career=career
            )
        else:
            resume_context = "No resume data provided. Assume the user is a beginner with minimal prior knowledge unless specified in '{details}'."

        # Craft a highly detailed prompt with resume integration
        prompt = f"""
        You are an expert career advisor with deep knowledge of industry trends, skill development, and professional growth strategies as of March 24, 2025. Your task is to create a highly detailed, actionable, and personalized level-wise career roadmap for someone pursuing a career as a '{career}'. Use the following additional details (if provided) to further tailor the roadmap: '{details}'.

        {resume_context}

        The roadmap must:
        - Show a clear, logical progression from the user's current state (based on resume data or beginner if none provided) to an ambitious yet achievable end goal (expert or leadership role in {career}).
        - Be formatted as a step-by-step guide with 5-6 distinct levels of progression, each representing a meaningful career milestone.
        - Be practical, specific, and focused on concrete steps that the user can follow immediately.
        - Reflect current industry standards, emerging technologies, and job market demands for {career} as of March 24, 2025.
        - Use a professional yet encouraging tone to motivate the user while maintaining realism about timelines and effort required.

        Structure the roadmap exactly as follows, with no deviations in headings or subheadings. For each section, follow the detailed instructions provided:

        # Career Roadmap: {career}
        **Purpose**: Provide a comprehensive guide to help the user navigate their career journey from their current state to a position of expertise or leadership in {career}.

        ## Starting Point
        - If resume data is provided, describe the user's current qualifications, skills, and experience based on their resume (e.g., "Currently a Junior {career} with 2 years of experience in X, skilled in Y").
        - If no resume data is available, assume a beginner level and describe the minimum qualifications, skills, or experience needed to start (e.g., "No prior experience, basic understanding of X recommended").
        - Incorporate '{details}' if it specifies a starting point (e.g., years of experience, specific knowledge).
        - Specify any prerequisite tools or knowledge they already have or need to acquire.
        - Keep this section concise (2-4 sentences) but precise.

        ## End Goal
        - Define the ultimate career position or level of expertise (e.g., Senior {career}, {career} Director, industry-recognized expert).
        - If resume data includes a professional summary or experience, align the end goal with their aspirations or trajectory (e.g., if they’re a team lead, aim for a managerial role).
        - Explain why this is a meaningful goal, tying it to industry impact, salary potential, or leadership opportunities as of March 24, 2025.

        ## Level-by-Level Progression
        - Create 5-6 distinct levels, each representing a clear career stage (e.g., Beginner, Junior, Mid-level, Senior, Expert/Leader).
        - Name each level descriptively to reflect the user’s role or expertise at that stage, adjusting based on resume data (e.g., if they have 2 years of experience, start at "Junior" rather than "Beginner").
        - For each level, include the following subheadings with specific, actionable content:

        ### Level 1: [Descriptive Level Name]
        - **Skills to Acquire**: List 4-6 specific, industry-relevant skills. If resume data shows existing skills, focus on gaps or next-level competencies (e.g., if they know Python, suggest advanced libraries like Pandas). Avoid vague terms.
        - **Learning Resources**: Recommend 3-5 specific resources (e.g., "Coursera: Machine Learning by Andrew Ng," "Clean Code by Robert C. Martin"). Build on certifications or education from resume data if available. Include URLs or platforms where possible.
        - **Projects to Build**: Suggest 2-3 practical projects that extend their existing projects or skills from resume data (e.g., if they built a basic app, suggest "Enhance it with a database"). Provide detailed, implementable ideas aligned with {career}.
        - **Estimated Time Investment**: Provide a realistic timeframe (e.g., "3-6 months full-time" or "6-12 months part-time at 10 hours/week"), adjusted based on '{details}' (e.g., time availability) and their current experience level.
        - **Transition Signal**: Describe 1-2 measurable indicators of readiness for the next level (e.g., "Completed a project deployed online" or "Passed a certification exam listed in resources").

        ### Level 2: [Descriptive Level Name]
        - Follow the same structure, building on Level 1. Use resume data to ensure continuity (e.g., if they’re multilingual, suggest skills leveraging that).

        ### Level 3: [Descriptive Level Name]
        - Introduce intermediate skills and responsibilities. If resume data shows domain experience, tailor this level to deepen that expertise.

        ### Level 4: [Descriptive Level Name]
        - Focus on advanced skills and specialization. If certifications are present, suggest advanced ones or leadership prep.

        ### Level 5: [Descriptive Level Name]
        - Emphasize expert-level competencies or strategic roles. Align with resume summary if it indicates long-term goals.

        ### Level 6: [Descriptive Level Name] (Optional, include if logical)
        - Detail a leadership or niche expert role, reflecting resume aspirations or industry trends.

        ## Alternative Pathways
        - Describe 2-3 alternative specializations within {career} (e.g., for a Software Engineer: "DevOps Engineer," "AI Developer"). If resume data suggests interests (e.g., projects in AI), prioritize related paths.
        - For each pathway:
          - Provide a 2-3 sentence description and its appeal.
          - List 1-2 unique skills, building on resume skills if applicable.
          - Suggest 1 resource or project tied to their background.

        ## Common Obstacles and Solutions
        - Identify 3-4 challenges based on resume data and career stage (e.g., if they lack experience, "Breaking into the field"; if mid-level, "Skill stagnation").
        - For each:
          - Describe it in 1-2 sentences with empathy.
          - Offer 1-2 solutions tailored to their skills or experience (e.g., "Leverage your Python skills in open-source projects").

        ## Additional Guidelines
        - Use resume data as the primary personalization source, supplemented by '{details}'.
        - Reflect 2025 industry trends (e.g., AI tools, new frameworks) and job market demands for {career}.
        - Avoid generic advice; all recommendations must be specific to {career} and the user’s profile.
        - If '{details}' includes constraints (e.g., time, location), adjust accordingly.
        - Include at least one cutting-edge skill or tool relevant to {career} in 2025.
        - Aim for 1500-2000 words, balancing depth and clarity.

        Now, generate the roadmap following this exact structure and guidance.
        """
        
        # Generate response from Gemini
        response = model.generate_content(prompt)
        
        # Return the formatted text
        return response.text
    
    except Exception as e:
        st.error(f"Error generating roadmap with Gemini: {str(e)}")
        # Fallback to template-based method
        return generate_level_wise_roadmap(career)

# Level-wise roadmap function (as fallback)
def generate_level_wise_roadmap(career):
    career_lower = career.lower()
    
    # Check if we have a template for this career
    if career_lower in CAREER_TEMPLATES:
        template = CAREER_TEMPLATES[career_lower]
        
        roadmap = f"""
        # Career Roadmap: {career.title()}
        
        ## Starting Point
        {template['starting_point']}
        
        ## End Goal
        {template['end_goal']}
        
        ## Level-by-Level Progression
        """
        
        # Add each progression level
        for level_data in template['progression_levels']:
            roadmap += f"""
            ### {level_data['level']}
            - **Skills to Acquire**: {', '.join(level_data['skills_to_acquire'])}
            - **Learning Resources**: {', '.join(level_data['learning_resources'])}
            - **Projects to Build**: {', '.join(level_data['projects'])}
            - **Estimated Time Investment**: {level_data['time_estimate']}
            """
        
        roadmap += """
        ## Alternative Pathways
        - Management track: Team Lead → Engineering Manager → Director
        - Specialist track: Focus on becoming an expert in a specific domain or technology
        - Entrepreneurial track: Building your own products or consulting
        
        ## Common Obstacles and Solutions
        - **Skill plateaus**: Overcome by finding new challenging projects and peer learning
        - **Impostor syndrome**: Join communities, contribute to open source, document achievements
        - **Technology changes**: Dedicate time for continuous learning and fundamentals
        - **Career transitions**: Build projects in target area, network with professionals, get relevant certifications
        """
        
    else:
        # Generic roadmap template
        roadmap = f"""
        # Career Roadmap: {career.title()}
        
        ## Starting Point
        Basic understanding of the field and foundational skills
        
        ## End Goal
        Expert/leadership position with strategic impact in the organization
        
        ## Level-by-Level Progression
        
        ### Level 1: Foundation Building
        - **Skills to Acquire**: Core technical fundamentals, industry terminology, basic tools
        - **Learning Resources**: Online introductory courses, beginner books, tutorials
        - **Projects to Build**: Simple applications or analyses that demonstrate basic skills
        - **Estimated Time Investment**: 3-6 months
        
        ### Level 2: Professional Beginner
        - **Skills to Acquire**: Standard methodologies, common tools, teamwork skills
        - **Learning Resources**: Intermediate courses, professional documentation, community forums
        - **Projects to Build**: Contributions to team projects, personal portfolio pieces
        - **Estimated Time Investment**: 6-12 months
        
        ### Level 3: Competent Professional
        - **Skills to Acquire**: Advanced techniques, efficiency skills, specialization foundations
        - **Learning Resources**: Advanced courses, specialized books, industry conferences
        - **Projects to Build**: End-to-end projects with moderate complexity
        - **Estimated Time Investment**: 1-2 years
        
        ### Level 4: Proficient Specialist
        - **Skills to Acquire**: Deep domain expertise, leadership fundamentals, mentoring skills
        - **Learning Resources**: Expert-level content, leadership training, industry networks
        - **Projects to Build**: Complex projects, team leadership opportunities
        - **Estimated Time Investment**: 2-3 years
        
        ### Level 5: Expert / Leader
        - **Skills to Acquire**: Strategic thinking, advanced leadership, innovation practices
        - **Learning Resources**: Executive education, industry leadership forums, peer networks
        - **Projects to Build**: Strategic initiatives, organizational improvements, mentorship programs
        - **Estimated Time Investment**: 3-5+ years
        
        ## Alternative Pathways
        - Management track: Team Lead → Department Manager → Executive
        - Technical specialist: Deep expertise in specific technologies or domains
        - Entrepreneur/Consultant: Independent business based on acquired expertise
        
        ## Common Obstacles and Solutions
        - **Skill plateaus**: Seek new challenges, cross-train in adjacent areas
        - **Impostor syndrome**: Document achievements, seek feedback, mentor others
        - **Work-life balance**: Establish boundaries, prioritize health, improve efficiency
        - **Technology changes**: Dedicate time to continuous learning, focus on fundamentals
        """
    
    return roadmap

# Function to search arXiv for papers
@st.cache_data
def search_arxiv(query):
    """Search Arxiv for industry research papers"""
    try:
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results=5"
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.text)
        
        # Extract entries
        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            authors = [author.find('{http://www.w3.org/2005/Atom}name').text 
                      for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
            link = entry.find('{http://www.w3.org/2005/Atom}id').text
            
            papers.append({
                "title": title,
                "authors": authors,
                "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                "link": link
            })
        
        return papers
    except Exception as e:
        st.error(f"Error searching arXiv: {str(e)}")
        return []

# NEW FUNCTION: Fetch job listings using SERP API
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_job_listings(career, location=None):
    """Fetch job listings using SerpAPI for a given career and optional location"""
    try:
        if not SERP_API_KEY:
            return {"error": "SERP API key not configured"}
        
        params = {
            "engine": "google_jobs",
            "q": career,
            "api_key": SERP_API_KEY,
            "hl": "en",
        }
        
        if location:
            params["location"] = location
        
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()
        
        job_results = []
        if "jobs_results" in data:
            for job in data["jobs_results"][:10]:  # Limit to 10 results
                job_results.append({
                    "title": job.get("title", "No title"),
                    "company_name": job.get("company_name", "Unknown company"),
                    "location": job.get("location", "No location"),
                    "description": job.get("description", "No description"),
                    "date_posted": job.get("detected_extensions", {}).get("posted_at", "Unknown"),
                    "salary": job.get("detected_extensions", {}).get("salary", "Not specified"),
                    "job_id": job.get("job_id", "")
                })
        
        return job_results
    except Exception as e:
        return {"error": str(e)}

# NEW FUNCTION: Fetch industry reports using SERP API
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_industry_reports(career, industry=None):
    """Fetch industry reports and articles using SerpAPI"""
    try:
        if not SERP_API_KEY:
            return {"error": "SERP API key not configured"}
        
        search_query = f"{career} industry report"
        if industry:
            search_query += f" {industry}"
            
        params = {
            "engine": "google",
            "q": search_query,
            "api_key": SERP_API_KEY,
            "num": 10,
            "tbm": "nws"  # This targets news results
        }
        
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()
        
        reports = []
        if "news_results" in data:
            for result in data["news_results"]:
                reports.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "source": result.get("source", ""),
                    "date": result.get("date", ""),
                    "snippet": result.get("snippet", "")
                })
        
        return reports
    except Exception as e:
        return {"error": str(e)}

# NEW FUNCTION: Fetch salary data using SERP API
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_salary_data(career, location=None):
    """Fetch salary information using SerpAPI for a given career"""
    try:
        if not SERP_API_KEY:
            return {"error": "SERP API key not configured"}
        
        search_query = f"{career} salary"
        if location:
            search_query += f" in {location}"
            
        params = {
            "engine": "google",
            "q": search_query,
            "api_key": SERP_API_KEY,
        }
        
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract knowledge graph if available (often contains salary info)
        if "knowledge_graph" in data:
            return {
                "source": "knowledge_graph",
                "data": data["knowledge_graph"]
            }
        
        # Otherwise extract some organic results that might contain salary info
        elif "organic_results" in data:
            relevant_results = []
            for result in data["organic_results"][:5]:
                if any(keyword in result.get("title", "").lower() for keyword in 
                      ["salary", "pay", "compensation", "income"]):
                    relevant_results.append({
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", "")
                    })
            return {
                "source": "organic_results",
                "data": relevant_results
            }
        
        return {"error": "No salary data found"}
    except Exception as e:
        return {"error": str(e)}

# Function to analyze job and industry data using Gemini
def analyze_job_market_data(career, job_listings, industry_reports, salary_data):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create a structured summary of job listings
        job_listings_summary = "No job listings data available."
        if isinstance(job_listings, list) and job_listings:
            job_listings_summary = "### Recent Job Listings Summary:\n"
            for job in job_listings[:5]:  # Limit analysis to first 5 listings
                job_listings_summary += f"- {job.get('title')} at {job.get('company_name')}, {job.get('location')}\n"
                job_listings_summary += f"  Salary: {job.get('salary', 'Not specified')}\n"
        
        # Create a structured summary of industry reports
        industry_reports_summary = "No industry reports data available."
        if isinstance(industry_reports, list) and industry_reports:
            industry_reports_summary = "### Recent Industry Reports:\n"
            for report in industry_reports[:5]:
                industry_reports_summary += f"- {report.get('title')} ({report.get('date')})\n"
                industry_reports_summary += f"  Source: {report.get('source')}\n"
                if 'snippet' in report:
                    industry_reports_summary += f"  Summary: {report.get('snippet')}\n"
        
        # Create a structured summary of salary data
        salary_summary = "No salary data available."
        if isinstance(salary_data, dict) and "data" in salary_data:
            if salary_data["source"] == "knowledge_graph" and salary_data["data"]:
                kg = salary_data["data"]
                salary_summary = "### Salary Information:\n"
                if "salary" in kg:
                    salary_summary += f"- Average Salary: {kg.get('salary')}\n"
                if "description" in kg:
                    salary_summary += f"- Description: {kg.get('description')}\n"
            elif salary_data["source"] == "organic_results" and salary_data["data"]:
                salary_summary = "### Salary Information from Various Sources:\n"
                for result in salary_data["data"][:3]:
                    salary_summary += f"- {result.get('title')}\n"
                    if 'snippet' in result:
                        salary_summary += f"  {result.get('snippet')}\n"
        
        # Craft the prompt
        prompt = f"""
        Analyze the following job market data for {career} professionals and provide a comprehensive market analysis:

        {job_listings_summary}

        {industry_reports_summary}

        {salary_summary}

        Please create a detailed job market analysis that includes:

        1. Current demand analysis (high, medium, or low) with supporting evidence
        2. Salary expectations for different experience levels (entry, mid, senior)
        3. Geographic areas with the most opportunities
        4. Industry trends and future outlook (next 3-5 years)
        5. Key skills employers are currently seeking
        6. Recommendations for job seekers in this field

        Format the response in clear, concise markdown with headings and bullet points.
        """
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        st.error(f"Error analyzing job market data: {str(e)}")
        return f"""
        # Job Market Analysis: {career}
        
        ## Current Demand
        - Unable to perform real-time analysis, but you can research current demand on job boards like LinkedIn, Indeed, and Glassdoor
        
        ## Salary Expectations
        - Entry-level: Research current rates for your location
        - Mid-level: Research current rates for your location
        - Senior-level: Research current rates for your location
        
        ## Key Skills in Demand
        - Research current in-demand skills for {career} roles in your target location
        
        *Note: For accurate, current job market analysis, consider checking industry reports from sources like LinkedIn's Workforce Report, Bureau of Labor Statistics, and industry-specific professional organizations.*
        """

# Function to get job trends using Gemini
def get_job_market_trends(career):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Provide a concise analysis of the current job market trends for {career} professionals.
        Include:
        1. Current demand (high, medium, low)
        2. Salary range expectations at different career levels (entry, mid, senior)
        3. Industries with highest demand
        4. Geographic areas with most opportunities
        5. Future outlook (next 3-5 years)
        
        Format the response in markdown with bullet points for easy reading.
        Keep it factual and data-focused.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error getting job market trends: {str(e)}")
        return "Unable to retrieve job market trends at this time."

def recommend_level_specific_resources(career, level, specific_skills="", industry_preference=None, time_commitment=None):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')

        specific_skills_context = f"with a focus on these specific skills/technologies: {specific_skills}" if specific_skills else ""

        industry_context = ""
        if industry_preference and len(industry_preference) > 0:
            industry_context = f"The resources should be relevant to these industries: {', '.join(industry_preference)}."

        time_context = ""
        if time_commitment:
            time_commitment = str(time_commitment).strip()  # Remove extra spaces
            if "Low" in time_commitment:
                time_context = "The user has **limited study time (0 to 5 hours per week)**, so resources should be concise and efficient."
            elif "Medium" in time_commitment:
                time_context = "The user has **moderate study time (5 to 10 hours per week)**, so resources should be balanced in depth and duration."
            elif "High" in time_commitment:
                time_context = "The user has **significant study time (10 to 20 hours per week)** and can handle in-depth courses and projects."
            elif "Full-time" in time_commitment:
                time_context = "The user is a **full-time learner**, so resources can be highly detailed and comprehensive."

        prompt = f"""
        Recommend specific learning resources for someone pursuing a career as a {career} {specific_skills_context}
        who is currently at the {level} level.

        {industry_context}
        {time_context}

        Include:
        1. Highly specific online courses that target their career path and skills
           - Provide actual course names, not just platforms
           - Focus on their specific skills: {specific_skills if specific_skills else "core skills for their level"}

        2. Books tailored to {career} professionals at the {level} level
           - Include author names and why each book is valuable

        3. YouTube channels, video series, or podcasts specifically about {career} {specific_skills_context if specific_skills else ""}
           - Name specific channels/series, not just general recommendations

        4. Professional communities or forums focused on {career} {specific_skills_context if specific_skills else ""}
           - Include both online communities and potential in-person networking opportunities

        5. Practical projects that specifically build skills in {specific_skills if specific_skills else f"core {career} competencies"} at the {level} level
           - Projects should be detailed enough to start working on immediately
           - Projects should demonstrate skills relevant to {', '.join(industry_preference) if industry_preference and len(industry_preference) > 0 else "the industry"}

        Format your response in markdown with clear headings and brief descriptions for each resource.
        Be extremely specific and targeted - avoid generic recommendations.
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        st.error(f"Error generating learning resources: {str(e)}")
        return "Could not generate learning resources at this time."


def generate_enhanced_market_insights(career, location=None, industry_preference=None):
    """
    Generate comprehensive market insights by combining SERP API data with Gemini analysis
    """
    try:
        # 1. Collect data from various SERP API endpoints
        job_listings = fetch_job_listings(career, location)
        
        # Use the first selected industry for targeted industry reports if available
        industry_query = industry_preference[0] if industry_preference and len(industry_preference) > 0 else None
        industry_reports = fetch_industry_reports(career, industry_query)
        
        # Get salary information
        salary_data = fetch_salary_data(career, location)
        
        # Check if we have meaningful data to analyze
        has_job_data = isinstance(job_listings, list) and len(job_listings) > 0
        has_reports = isinstance(industry_reports, list) and len(industry_reports) > 0
        has_salary = isinstance(salary_data, dict) and "data" in salary_data
        
        if not (has_job_data or has_reports or has_salary):
            # If no data from SERP API, use Gemini for general insights
            return get_job_market_trends(career)
        
        # 2. Process and format the collected data for Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Format job listings data
        job_data_formatted = "### Job Listings Data:\n"
        if has_job_data:
            for i, job in enumerate(job_listings[:8]):  # Limit to 8 listings for analysis
                job_data_formatted += f"**Job {i+1}:** {job.get('title')} at {job.get('company_name')}\n"
                job_data_formatted += f"- Location: {job.get('location', 'Not specified')}\n"
                job_data_formatted += f"- Posted: {job.get('date_posted', 'Not specified')}\n"
                job_data_formatted += f"- Salary: {job.get('salary', 'Not specified')}\n"
                job_data_formatted += f"- Description Excerpt: {job.get('description', 'No description')[:200]}...\n\n"
        else:
            job_data_formatted += "No specific job listing data available.\n"
        
        # Format industry reports data
        industry_data_formatted = "### Industry Reports and News:\n"
        if has_reports:
            for i, report in enumerate(industry_reports[:5]):
                industry_data_formatted += f"**Report {i+1}:** {report.get('title')}\n"
                industry_data_formatted += f"- Source: {report.get('source')} ({report.get('date', 'No date')})\n"
                industry_data_formatted += f"- Summary: {report.get('snippet', 'No summary available')}\n\n"
        else:
            industry_data_formatted += "No specific industry report data available.\n"
        
        # Format salary data
        salary_data_formatted = "### Salary Information:\n"
        if has_salary:
            if salary_data["source"] == "knowledge_graph" and salary_data["data"]:
                kg = salary_data["data"]
                if "salary" in kg:
                    salary_data_formatted += f"- Average Salary: {kg.get('salary')}\n"
                if "description" in kg:
                    salary_data_formatted += f"- Context: {kg.get('description')}\n"
                if "attributes" in kg:
                    salary_data_formatted += "- Additional Information:\n"
                    for k, v in kg.get("attributes", {}).items():
                        salary_data_formatted += f"  - {k}: {v}\n"
            elif salary_data["source"] == "organic_results" and salary_data["data"]:
                for i, result in enumerate(salary_data["data"][:3]):
                    salary_data_formatted += f"**Source {i+1}:** {result.get('title')}\n"
                    salary_data_formatted += f"- Information: {result.get('snippet', 'No details available')}\n\n"
        else:
            salary_data_formatted += "No specific salary data available.\n"
        
        # 3. Create a comprehensive prompt for Gemini to analyze the data
        prompt = f"""
        As a career insights expert, analyze this market data for {career} professionals 
        {f'in {location}' if location else ''} 
        {f'within the {industry_query} industry' if industry_query else ''}:

        {job_data_formatted}

        {industry_data_formatted}

        {salary_data_formatted}

        Based on this data, provide a comprehensive market analysis that includes:

        # {career} Job Market Analysis
        
        ## Current Demand
        [Analyze the current demand level (high, medium, low) with specific evidence from the data]
        
        ## Salary Insights
        [Provide salary ranges for entry, mid, and senior levels, with specific data references]
        
        ## Location Hotspots
        [Identify geographic areas with the most opportunities based on the data]
        
        ## Key Skills in Demand
        [Extract and list the most frequently mentioned skills from job descriptions]
        
        ## Industry Trends
        [Analyze current trends and future outlook for the next 3-5 years]
        
        ## Career Advancement Tips
        [Provide 3-4 specific, actionable recommendations for job seekers]

        Your analysis should reference the specific data provided where possible. If data is limited in any section, 
        provide the best insights possible based on the available information, without making up specific statistics.
        
        Format everything in clean, professional markdown.
        """
        
        # 4. Generate the enhanced analysis using Gemini
        response = model.generate_content(prompt)
        enhanced_analysis = response.text
        
        # 5. Add a download data section to the analysis
        enhanced_analysis += f"""
        
        ## Data Sources
        
        This analysis is based on real-time job market data retrieved on {datetime.datetime.now().strftime('%Y-%m-%d')}. 
        For the most current information, we recommend checking job boards and industry reports directly.
        """
        
        return enhanced_analysis
    
    except Exception as e:
        st.error(f"Error generating enhanced market insights: {str(e)}")
        # Fallback to basic insights
        return get_job_market_trends(career)

def generate_fallback_resources(career, level, specific_skills="", industry_preference=None):
    """Generate fallback learning resources when API is unavailable"""
    
    career_lower = career.lower()
    
    # Start building detailed fallback resources
    resources = f"""
    # Learning Resources for {career} at {level} Level
    
    ## Online Courses
    """
    
    # Add course recommendations based on career type
    if "data" in career_lower or "analy" in career_lower or "scien" in career_lower:
        resources += """
    - **Coursera:** IBM Data Science Professional Certificate or Google Data Analytics Certificate
    - **Udemy:** Python for Data Science and Machine Learning Bootcamp
    - **edX:** Harvard's Data Science Professional Certificate
    """
    elif "software" in career_lower or "develop" in career_lower or "engineer" in career_lower:
        resources += """
    - **Coursera:** Meta Backend/Frontend Developer Professional Certificate
    - **Udemy:** The Complete Web Developer Bootcamp or The Complete 2023 Software Engineering Bootcamp
    - **Codecademy:** Pro membership with career paths for software engineering
    """
    elif "design" in career_lower or "ux" in career_lower or "ui" in career_lower:
        resources += """
    - **Coursera:** Google UX Design Professional Certificate
    - **Udemy:** UI/UX Design with Figma Complete Course
    - **LinkedIn Learning:** Becoming a UX Designer
    """
    else:
        resources += f"""
    - **Coursera:** Look for certificates specific to {career}
    - **Udemy:** Search for comprehensive bootcamps in {career}
    - **LinkedIn Learning:** Essential training courses in {career}
    """
    
    # Add specific skills focus if provided
    if specific_skills:
        resources += f"""
    - **Specialized Skills:** Look for courses specifically teaching {specific_skills}
    """
    
    # Add book recommendations
    resources += f"""
    
    ## Books & Reading Materials
    """
    
    # Career-specific book recommendations
    if "data" in career_lower or "analy" in career_lower or "scien" in career_lower:
        resources += """
    - "Python for Data Analysis" by Wes McKinney
    - "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron
    - "Storytelling with Data" by Cole Nussbaumer Knaflic
    """
    elif "software" in career_lower or "develop" in career_lower or "engineer" in career_lower:
        resources += """
    - "Clean Code" by Robert C. Martin
    - "The Pragmatic Programmer" by Andrew Hunt and David Thomas
    - "Designing Data-Intensive Applications" by Martin Kleppmann
    """
    elif "design" in career_lower or "ux" in career_lower or "ui" in career_lower:
        resources += """
    - "Don't Make Me Think" by Steve Krug
    - "The Design of Everyday Things" by Don Norman
    - "Refactoring UI" by Adam Wathan & Steve Schoger
    """
    else:
        resources += f"""
    - Look for recent, highly-rated books specific to {career}
    - Find industry publications and journals related to {career}
    """
    
    # Add community recommendations
    resources += f"""
    
    ## Communities & Networking
    """
    
    # Career-specific community recommendations
    if "data" in career_lower or "analy" in career_lower or "scien" in career_lower:
        resources += """
    - **Kaggle:** Join competitions and discussion forums
    - **Reddit:** r/datascience, r/MachineLearning, r/dataanalysis
    - **Discord:** Data Science communities like DAIR or The Data Den
    """
    elif "software" in career_lower or "develop" in career_lower or "engineer" in career_lower:
                resources += """
    - **GitHub:** Contribute to open-source projects
    - **Reddit:** r/programming, r/webdev, r/cscareerquestions
    - **Discord:** Programming communities like The Coding Den or Code Support
    """
    elif "design" in career_lower or "ux" in career_lower or "ui" in career_lower:
        resources += """
    - **Dribbble:** Share work and get inspiration
    - **Behance:** Build your portfolio and see others' work
    - **Reddit:** r/UXDesign, r/web_design

    """
    else:
        resources += f"""
    - **LinkedIn Groups:** Find groups specific to {career}
    - **Reddit:** Look for subreddits focused on {career}
    - **Slack/Discord:** Search for professional communities in {career}
    """  
    resources += f"""
    
    ## Recommended Projects
    """
    
    # Career-specific project recommendations
    if "data" in career_lower or "analy" in career_lower or "scien" in career_lower:
        resources += """
    - Build a data dashboard for public datasets
    - Create a predictive model for a problem you're interested in
    - Perform exploratory data analysis on an industry dataset
    """
    elif "software" in career_lower or "develop" in career_lower or "engineer" in career_lower:
        resources += """
    - Develop a personal portfolio website
    - Create a web application with authentication and database
    - Build an API that solves a specific problem
    """
    elif "design" in career_lower or "ux" in career_lower or "ui" in career_lower:
        resources += """
    - Redesign the UI of an existing application
    - Create a complete UX case study
    - Design a mobile app interface for a specific user problem
    """
    else:
        resources += f"""
    - Build a portfolio showcasing your {career} skills
    - Create a project that demonstrates your expertise in {specific_skills if specific_skills else f"{career}"}
    - Document your learning journey in a blog or case study
    """
    
    # Add industry-specific recommendations if provided
    if industry_preference and len(industry_preference) > 0:
        industries = ', '.join(industry_preference)
        resources += f"""
    
    ## Industry-Specific Focus: {industries}
    - Research key players and trends in the {industries} industry
    - Target your projects to solve problems specific to {industries}
    - Look for specialized communities focused on {career} in {industries}
    """
    
    return resources  

with st.sidebar:
    st.subheader("Career Details")
    
    # Career input
    career_path = st.text_input("Enter your target career path:", "Data Scientist")
    
    # Current level selection
    level_options = [
        "Beginner (getting started)",
        "Junior (0-2 years experience)",
        "Mid-level (2-5 years experience)",
        "Senior (5+ years experience)",
        "Leadership/Management"
    ]
    # current_level = st.selectbox("Your current level:", level_options)
    
    # Specific skills (optional)
    specific_skills = st.text_area("Specific skills or technologies of interest:", 
                                   placeholder="E.g., Python, TensorFlow, SQL, cloud computing...")
    
    # Industry preferences (multi-select)
    industry_options = [
        "Technology", "Healthcare", "Finance", "Education", 
        "Manufacturing", "Retail", "Media", "Government", 
        "Non-profit", "Energy", "Transportation", "Entertainment"
    ]
    industry_preference = st.multiselect("Target industries:", industry_options)
    
    # Location preference
    location = st.text_input("Your preferred location:", 
                            placeholder="City, state, or country")
    
    # Time commitment 
    time_options = [
        "Low (0 to 5 hours per week)",
        "Medium (5 to 10 hours per week)",
        "High (10 to 20 hours per week)",
        "Full-time learner"
    ]
    time_commitment = st.selectbox("Weekly time commitment for learning:", time_options)
    
    # Advanced options expander
    with st.expander("Additional Information (Optional)"):
        background = st.text_area("Your educational/professional background:", 
                                placeholder="Degrees, certifications, previous roles...")
        goals = st.text_area("Career goals and aspirations:", 
                            placeholder="Where do you want to be in 5 years?")
        learning_style = st.selectbox("Your learning style preference:", 
                                    ["Visual", "Reading/Writing", "Interactive/Hands-on", "Mixed"])
    
    generate_button = st.button("Generate Career Roadmap", type="primary")

# Main content area tabs
tabs = st.tabs(["Career Roadmap", "Learning Resources", "Market Insights", "Research Papers"])

# If generate button is clicked
if generate_button or "roadmap_generated" in st.session_state:
    # Store state to prevent re-generation on refresh
    if generate_button:
        st.session_state.roadmap_generated = True
        st.session_state.career_path = career_path
    
    with st.spinner("Generating your personalized career roadmap..."):
        resume_data = st.session_state.get('parsed_resume',None)

        if GOOGLE_API_KEY:
            try:
                # Create detailed prompt with all user inputs
                # details = ""
                # if specific_skills:
                #     details += f"Focusing on these specific skills: {specific_skills}. "
                # if industry_preference:
                #     details += f"For these target industries: {', '.join(industry_preference)}. "
                # if location:
                #     details += f"For someone based in {location}. "
                # if "background" in st.session_state and st.session_state.background:
                #     details += f"Based on this background: {st.session_state.background}. "
                # if "goals" in st.session_state and st.session_state.goals:
                #     details += f"With these career goals: {st.session_state.goals}. "
                # if "learning_style" in st.session_state and st.session_state.learning_style:
                #     details += f"With a preference for {st.session_state.learning_style} learning. "
                details = f"""
                Focusing on skills: {specific_skills}.
                For industries: {', '.join(industry_preference)}.
                Location: {location}.
                Time commitment: {time_commitment}.
                """

                
                # Generate roadmap with Gemini
                roadmap_content = generate_gemini_roadmap(career_path, details,resume_data)
                
                st.session_state.roadmap_content = roadmap_content
                
                # Generate level-specific resources based on current level
                # level_mapping = {
                #     "Beginner (getting started)": "Foundational Level",
                #     "Junior (0-2 years experience)": "Junior Level",
                #     "Mid-level (2-5 years experience)": "Mid-Level Professional",
                #     "Senior (5+ years experience)": "Senior Level",
                #     "Leadership/Management": "Leadership Level"
                # }
                
                # mapped_level = level_mapping.get(current_level, "Appropriate Level")
                
                # Generate resources content
                resources_content = recommend_level_specific_resources(
                    career_path, 
                    # mapped_level, 
                    specific_skills,
                    industry_preference,
                    time_commitment
                )
                st.session_state.resources_content = resources_content
                
                # Generate job market insights
                try:
                    # Fetch job listings with location if provided
                    job_listings = fetch_job_listings(career_path, location if location else None)
                    
                    # Fetch industry reports
                    industry_query = industry_preference[0] if industry_preference else None
                    industry_reports = fetch_industry_reports(career_path, industry_query)
                    
                    # Fetch salary data
                    salary_data = fetch_salary_data(career_path, location if location else None)
                    
                    # Analyze market data
                    market_insights = generate_enhanced_market_insights(career_path, location if location else None , industry_preference)
                    st.session_state.market_insights = market_insights
                except Exception as e:
                    # Fallback for market insights
                    st.session_state.market_insights = get_job_market_trends(career_path)
                
                # Fetch research papers
                try:
                    papers = search_arxiv(f"{career_path} {specific_skills if specific_skills else ''}")
                    st.session_state.papers = papers
                except Exception as e:
                    st.session_state.papers = []
                
            except Exception as e:
                st.error(f"Error generating with Gemini: {str(e)}")
                # Fallback to template-based approach
                st.session_state.roadmap_content = generate_level_wise_roadmap(career_path)
                st.session_state.resources_content = generate_fallback_resources(
                    career_path, 
                    # level_mapping.get(current_level, "Appropriate Level"),
                    specific_skills,
                    industry_preference
                )
                st.session_state.market_insights = get_job_market_trends(career_path)
        else:
            # No API key - use template-based approach
            st.session_state.roadmap_content = generate_level_wise_roadmap(career_path)
            st.session_state.resources_content = generate_fallback_resources(
                career_path, 
                specific_skills,
                industry_preference
            )
            st.session_state.market_insights = get_job_market_trends(career_path)
    
    # Display content in tabs
    with tabs[0]:  # Career Roadmap tab
        st.markdown(st.session_state.roadmap_content)
        
        # Download button
        roadmap_text = st.session_state.roadmap_content
        st.download_button(
            "Download Roadmap",
            roadmap_text,
            file_name=f"{career_path.replace(' ', '_')}_roadmap.md",
            mime="text/markdown"
        )
    
    with tabs[1]:  # Learning Resources tab
        st.markdown(st.session_state.resources_content)
        
        # Download button
        resources_text = st.session_state.resources_content
        st.download_button(
            "Download Resources",
            resources_text,
            file_name=f"{career_path.replace(' ', '_')}_resources.md",
            mime="text/markdown"
        )
    
    with tabs[2]:  # Market Insights tab
        st.markdown(f"Data Retrieved on {datetime.now().strftime('%Y-%m-%d')}*")
        st.markdown(st.session_state.market_insights)

        market_text = st.session_state.market_insights
        st.download_button("Download market insights",market_text,file_name=f"{career_path.replace(' ', '_')}_market_insights.md",
                           mime="text/markdown")
    

    
    with tabs[3]:  # Research Papers tab
        if hasattr(st.session_state, 'papers') and st.session_state.papers:
            for i, paper in enumerate(st.session_state.papers):
                with st.expander(f"{paper['title']}"):
                    st.write(f"**Authors:** {', '.join(paper['authors'])}")
                    st.write(f"**Summary:** {paper['summary']}")
                    st.write(f"**Link:** {paper['link']}")
        else:
            st.info(f"No recent research papers found for {career_path}. Try adjusting your search terms or check academic databases directly.")

else:
    # Initial state - show welcome information
    with tabs[0]:
        st.markdown("""
        # Welcome to the Career Roadmap Generator! 🚀
        
        This tool helps you create a personalized career progression plan based on your target career path, 
        current level, and specific interests.
        
        ## How to use this tool:
        
        1. Fill in your career details in the sidebar
        2. Click the "Generate Career Roadmap" button
        3. Explore your personalized roadmap, resources, and insights in the tabs above
        
        ## What you'll get:
        
        - **Career Roadmap**: A detailed progression plan from your current level to expert
        - **Learning Resources**: Personalized recommendations for courses, books, and projects
        - **Market Insights**: Current job market analysis and salary information
        - **Research Papers**: Recent academic research in your field
        
        Get started by entering your details in the sidebar! →
        """)

# Footer
st.markdown("---")
