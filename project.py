import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

#Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wikipedia_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
wikipedia=WikipediaQueryRun(api_wrapper=wikipedia_wrapper)
search=DuckDuckGoSearchRun(name="Search")


# Streamlit App
st.title("🔎 Gen AI Search Engine")
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter Groq API key",type="password")

if 'messages' not in st.session_state:
    st.session_state["messages"]=[
        {"role":"Assistant","content":"Hi, How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="Example:What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    llm=ChatGroq(api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[search,arxiv,wikipedia]
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    response=search_agent.run(st.session_state.messages)
    st.session_state["messages"].append({"role":"Assistant","content":response})
    st.write(response)
    