import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_core.globals import set_debug


set_debug(True)

st.set_page_config(page_title="Gemini Python Agent", layout="wide")
st.title("🚀 Gemini Python Code Agent")

# gemini_model = os.getenv("gemini_model") or "gemini-1.5-pro"
# gemini_api_key = os.getenv("gemini_api_key")

with st.sidebar:
    st.title("provide your GEMINI API key")
    gemini_api_key = st.text_input("enter key here", type="password")

if not gemini_api_key:
    st.info("please provide your gemini api key to continue")

    st.stop()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.2,
    api_key=gemini_api_key,
)


tools = [
    DuckDuckGoSearchRun(
        name="search",
        description="Search the web for Python libraries, APIs, or examples"
    )
]


prompt = PromptTemplate.from_template("""
You are a senior Python engineer using the ReAct pattern.

You have access to the following tools:
{tools}

Tool names:
{tool_names}

Instructions:
- You MAY think internally using Thought / Action / Observation
- Use tools ONLY if required
- When you are done, you MUST respond EXACTLY in this format:

Final Answer:
<only valid runnable Python code>

Rules for Final Answer:
- Python code only
- No markdown
- No explanations
- No backticks

User task:
{input}

{agent_scratchpad}
""")


agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=2,          
    max_execution_time=20,  
)


user_input = st.text_area(
    "Describe the Python program you want:",
    height=160,
)

if st.button("Generate Python Code"):
    try:
        result = agent_executor.invoke({"input": user_input})
        st.code(result["output"], language="python")
    except Exception as e:
        st.error(e)
