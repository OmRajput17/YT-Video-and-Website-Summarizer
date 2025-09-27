import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
from dotenv import load_dotenv

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
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(youtube_url=url, add_video_info = True)
                else:
                    loader = UnstructuredURLLoader(urls=[url], ssl_verified = False, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                
                docs = loader.load()

                #### Chain For Summarization
                chain = load_summarize_chain(
                    llm = llm, 
                    chain_type="stuff",
                    prompt = prompt,
                )

                summary = chain.run(docs)

                st.success(summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
            