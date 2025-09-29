import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re

load_dotenv()

### StreamLit App

st.set_page_config(page_title="LangChain :Summarize Text from YT or Website")
st.title("LangChain :Summarize Text from YT or Website")
st.subheader("Summarize URL")


### Creating URL Fields
url = st.text_input("URL", label_visibility="collapsed")

api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key = api_key, model = "gemma2-9b-it")

prompt_template = """
provide the summary of the following content in 500 words:
Content : {text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables= ['text']
)

### Helper Functions (replacements for broken loaders)

def load_youtube_transcript(url):
    """Replace YoutubeLoader with working transcript fetcher"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Extract video ID
        video_id_match = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
        if not video_id_match:
            raise Exception("Invalid YouTube URL")
        
        video_id = video_id_match.group(1)
        
        # Get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([item['text'] for item in transcript_list])
        
        return [Document(
            page_content=transcript_text,
            metadata={"source": url, "video_id": video_id}
        )]
    
    except ImportError:
        raise Exception("youtube-transcript-api not installed. Run: pip install youtube-transcript-api")
    
    except AttributeError:
        # If there's an attribute error, try alternative import
        try:
            import youtube_transcript_api
            transcript_list = youtube_transcript_api.YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = ' '.join([item['text'] for item in transcript_list])
            
            return [Document(
                page_content=transcript_text,
                metadata={"source": url, "video_id": video_id}
            )]
        except:
            raise Exception(
                "youtube-transcript-api package issue. Try:\n"
                "1. pip uninstall youtube-transcript-api\n"
                "2. pip install youtube-transcript-api\n"
                "3. Restart your Python environment"
            )

def load_website_content(url):
    """Replace UnstructuredURLLoader with working scraper"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    response = requests.get(url, headers=headers, timeout=15, verify=False)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove unwanted elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()
    
    # Get main content
    main_content = soup.find('article') or soup.find('main') or soup.find('body')
    text = main_content.get_text(separator='\n', strip=True)
    
    # Clean up
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = '\n'.join(lines)
    
    return [Document(
        page_content=text,
        metadata={"source": url}
    )]


if st.button("Summarize the Content from YT or Website"):
    ### Validates all the inputs
    if not url.strip():
        st.error("Please provide the Information")
    elif not validators.url(url):
        st.error("Please Enter a Valid URL")
    else:
        try:
            with st.spinner("Waiting..."):
                ### loading the website or yt video data
                if "youtube.com" in url or "youtu.be" in url:
                    docs = load_youtube_transcript(url)
                else:
                    docs = load_website_content(url)

                #### Chain For Summarization
                chain = load_summarize_chain(
                    llm = llm, 
                    chain_type="stuff",
                    prompt = prompt,
                )

                # Use invoke() instead of deprecated run()
                output = chain.invoke({"input_documents": docs})
                summary = output["output_text"]

                st.success(summary)
        except Exception as e:
            st.exception(f"Exception:{e}")