# CareerForge AI - README

## Overview
CareerForge AI is a Streamlit-powered application that provides AI-driven career guidance. It helps users by analyzing resumes, generating personalized career roadmaps, recommending learning resources, and providing real-time job market insights.

## Features
- **Resume Parsing**: Extracts structured information from uploaded PDF resumes.
- **Career Roadmaps**: AI-generated career progression plans based on skills and experience.
- **Learning Resources**: Curated courses, books, and projects tailored to career goals.
- **Job Market Insights**: Fetches live job listings, salary data, and industry trends.
- **Research Papers**: Searches and displays relevant academic papers from ArXiv.

## Technologies Used
- **Frontend**: Streamlit
- **AI & NLP**: Google Generative AI (Gemini API), LangChain, FAISS
- **Data Processing**: PyPDF2, RecursiveCharacterTextSplitter
- **APIs**: SERP API (job listings & salary data), ArXiv API (research papers)

## Installation
### Prerequisites
- Python 3.8+
- API Keys for Google Generative AI & SERP API

### Steps to Install and Run
1. Clone the repository:
   ```sh
   git clone https://github.com/TejaswiMahadev/CareerForge.git
   cd CareerForge
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up `.env` file with API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   SERPAPI_KEY=your_serpapi_key
   ```
4. Run the application:
   ```sh
   streamlit run app.py
   ```

## Usage
1. **Upload a Resume**: Upload a PDF file to extract and analyze career-related information.
2. **Generate Career Roadmap**: Enter career preferences and receive an AI-generated progression plan.
3. **View Learning Resources**: Get personalized recommendations for courses, books, and projects.
4. **Analyze Job Market Trends**: Fetch and review job listings, salary data, and industry insights.
5. **Explore Research Papers**: Search for relevant academic papers on career-related topics.

## Error Handling
- Prompts for missing API keys.
- Displays errors for resume parsing failures.
- Provides fallback options for API failures.

## Future Enhancements
- Advanced AI resume analysis.
- More job listing sources beyond SERP API.
- Interactive UI improvements.

## License
MIT License

## Contact
For support or collaboration, reach out to tejaswimahadev9@gmail.com.

