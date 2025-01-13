import pandas as pd
import os
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_groq import ChatGroq

# Load the CSV file
df = pd.read_csv('sample.csv')
# print(df.head())

# Set the API key
api_key = "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"
os.environ["GROQ_API_KEY"] = api_key  # Ensure this is correct
# print("API Key Set:", os.environ.get("GROQ_API_KEY"))  # Debugging line
# # Create the ChatGroq instance with the API key
llm = ChatGroq(api_key=api_key, temperature=0.5)

# # Create the agent executor with allow_dangerous_code set to True
agent_executer = create_csv_agent(llm, 'sample.csv', verbose=True, allow_dangerous_code=True)

# List of queries
queries = [
    "What is the stock price of AAPL?",
    "What is the performance of TSLA?",
    "What is the PE ratio of MSFT?",
    "What is the 52-week high of AAPL?",
    "Which stock has the highest dividend yield?",
    "What is the difference between the 52-week high and low for MSFT?",
    "Which stock has the highest price?",
    "What is the average stock price?"
]

# Collect responses
responses = []
for query in queries:
    response = agent_executer.invoke(query)
    responses.append({"query": query, "response": response})

# Convert responses to DataFrame
responses_df = pd.DataFrame(responses)

# Save to a new CSV file
responses_df.to_csv('responses.csv', index=False)

# Print the responses
print(responses_df)