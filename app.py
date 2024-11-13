# Import necessary modules with updated paths
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

import getpass
import os

#os.environ["GOOGLE_API_KEY"] = getpass.getpass("AIzaSyDbStyiiw5KCWJ5_YK4q6b6-MCz1jNdo2Q")

api_key = "AIzaSyBG5TCCSusvqD7sCnrZfz6TgHoRpJJIxXM"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=api_key,
    # other params...
)



# Define a prompt template
template = """You are a helpful assistant. Answer the following question:
{question}"""

# Create a PromptTemplate instance
prompt = PromptTemplate.from_template(template)

# Create an LLMChain instance
llm_chain = LLMChain(
    prompt=prompt,
    llm=llm
)

# Define a question to ask the model

question = '''Give the topic names for these and remove the note at the end of the response and display in the order and remove topic numberings and display them in a new line:
Topic 0: 0.040*"learning" + 0.027*"floor" + 0.027*"employees" + 0.027*"water",
Topic 1: 0.046*"cab" + 0.028*"using" + 0.028*"much" + 0.028*"service",
Topic 2: 0.004*"organisation" + 0.004*"ai" + 0.004*"explore" + 0.004*"taking",
Topic 3: 0.004*"people" + 0.004*"office" + 0.004*"day" + 0.004*"work",
Topic 4: 0.049*"organisation" + 0.049*"parking" + 0.025*"employees" + 0.025*"people",
Topic 5: 0.022*"food" + 0.022*"time" + 0.022*"also" + 0.022*"explore",
Topic 6: 0.062*"office" + 0.053*"work" + 0.036*"people" + 0.027*"day",
Topic 7: 0.004*"food" + 0.004*"also" + 0.004*"time" + 0.004*"taste",
Topic 8: 0.035*"whole" + 0.035*"get" + 0.035*"puzzle" + 0.035*"elevator",
Topic 9: 0.004*"office" + 0.004*"work" + 0.004*"people" + 0.004*"day"'''



# Generate a response using the LLMChain
response = llm_chain.run(question)

# Print the response
st.chat_message("assistant").write(response)
#print(response)
