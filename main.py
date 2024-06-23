import streamlit as st
import pandas as pd
import os
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.schema import HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Securely retrieving the OpenAI API key from Streamlit's secrets management
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# Load the CSV files
file_path1 = 'data/Original_data_with_more_rows.csv'
file_path2 = 'data/Expanded_data_with_more_features.csv'
data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)

# Load the RTF files
file_paths = [
    'data/gender_and_high_school_scores.txt.rtf',
    'data/race_gender_age_effect_test_scores.txt.rtf'
]

# Load and process the documents
documents = []
for file_path in file_paths:
    loader = TextLoader(file_path)
    documents.extend(loader.load())

# Initialize the language model
llm_1 = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)
llm_2 = ChatOpenAI(model="gpt-4", temperature=0, verbose=True)
llm_3 = ChatOpenAI(model="gpt-4", temperature=0, verbose=True)

# Create the Python REPL tool
python_repl_tool_1 = PythonREPLTool()
python_repl_tool_2 = PythonREPLTool()

# Create the embeddings for text files
embeddings = OpenAIEmbeddings()

# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create the vector store using FAISS
docsearch = FAISS.from_documents(texts, embeddings)

# Define the retriever
retriever = docsearch.as_retriever()

# Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm_1, 
    chain_type="stuff", 
    retriever=retriever
)

# Create the CSV agents
agent_executer1 = create_csv_agent(llm_2, file_path1, tools=[python_repl_tool_1], max_iterations=10, allow_dangerous_code=True, verbose=True, handle_parsing_errors=True)
agent_executer2 = create_csv_agent(llm_3, file_path2, tools=[python_repl_tool_2], max_iterations=10, allow_dangerous_code=True, verbose=True, handle_parsing_errors=True)

# Function to answer questions based on the data
def answer_question(agent_executer1, agent_executer2, question: str , qa):
    # Show the question, process the prompt
    print("Query: ", question)
    prompt = f"Here's a question: {question}. Answer it solely from the data file at hand. Do not mention that you do that. If you cannot find an answer, retrieve `I don't know the answer based on my data`"

    # First, try to get the answer from the first CSV agent
    answer_1 = agent_executer1.invoke(HumanMessage(prompt))
    output = answer_1['output']
    if output != "I don't know the answer based on my data." \
        and "iteration limit" not in output \
        and "time limit" not in output :
        return ("Answer: " + answer_1['output'])
        
    # If no answer from the first CSV agent, try the second CSV agent
    answer_2 = agent_executer2.invoke(HumanMessage(prompt))
    output = answer_2['output']
    if output != "I don't know the answer based on my data." \
        and "iteration limit" not in output \
        and "time limit" not in output:
        return ("Answer: " + answer_2['output'])

    # If no answer from either CSV agent, try to get the answer from the RTF data
    answer_3 = qa.invoke(prompt)
    output = answer_3['result']
    if output != "I don't know the answer based on my data." \
        and "iteration limit" not in output \
        and "time limit" not in output:
        return ("Answer: " + answer_3['result'])

    # If no answer from any data source, return the default response
    return "Answer: I don't know the answer based on my data."
    

## Streamlit interface
# Defining a title
st.title("AI-Powered Q&A App")

# Describing the app
st.markdown('In this app, I developed a solution for querying and extracting insights from a diverse set of data sources,' \
            'including two articles and two CSV files, as outlined in the provided exercise instructions.' \
            'The main goal was to enable asking free-form questions and obtaining answers that are substantiated by the data.' \
            'We want the app to be based on the data sources provided. Hence, if we ask an unrelated question, the app will indicate that it does not know the answer.')

question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question:
        answer = answer_question(agent_executer1, agent_executer2, question, qa)
        st.write(answer)
    else:
        st.write("Please enter a question to get an answer.")

# Define a list of example questions
example_questions = [
    "What are the effects of legislations surrounding emissions on the Australia coal market?",
    "What recent changes have been observed in gender differences in high school math and reading skills?",
    "What's the most common range of studying hours per week?",
    "How have societal attitudes towards gender roles influenced math achievement among students?",
    "What factors have been identified as influencing standardized test scores?",
    "How has the gender gap in educational attainment changed in recent years?",
    "What day is tomorrow?",
    "What's the average number of siblings for all observations?",
    "What's the max number of siblings for all observations?",
    "What's the mean score for writing?"
]

# Display the example questions in Markdown format
st.markdown("### Example Questions")
st.markdown("\n".join(f"- {question}" for question in example_questions))

print(answer_question(agent_executer1, agent_executer2, question, qa))
