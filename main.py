#############################################
# THIS IS THE APP VERSION WITHOUT STREAMLIT, FOR TESTING PURPOSES.
# You can write the question here and get the answer through the terminal.
#############################################

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
os.environ["OPENAI_API_KEY"] = "YOUR API KEY HERE!"

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
llm_validator = ChatOpenAI(model="gpt-4", temperature=0, verbose=True)  # Add a model for validation

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

# Function to validate the question using a language model
def is_valid_question_llm(question: str) -> bool:
    validation_prompt = f"Is the following text a valid, meaningful question? '{question}' Answer with 'Yes' or 'No'."
    validation_response = llm_validator.invoke(validation_prompt)
    return False if validation_response.content == 'No' else True

# Function to answer questions based on the data
def answer_question(agent_executer1, agent_executer2, question: str , qa):
    # Validate the question using the language model
    if not is_valid_question_llm(question):
        return "Invalid question. Please make sure your question is meaningful and ends with a question mark."
    
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
    
# Example query
# question = "What are the effects of legislations surrounding emissions on the Australia coal market?"
# question = "What recent changes have been observed in gender differences in high school math and reading skills?"
# question = "What's the most common range of studying hours per week?"
# question = "How have societal attitudes towards gender roles influenced math achievement among students?"
# question = "What factors have been identified as influencing standardized test scores?"
# question = "How has the gender gap in educational attainment changed in recent years?"
# question = "What day is tomorrow?"
# question = "What's the average number of siblings for all observations?"
# question = "What's the max number of siblings for all observations?"
# question = "what's the mean score for writing?"
question = "How has the gender gap in educational"

print(answer_question(agent_executer1, agent_executer2, question, qa))
