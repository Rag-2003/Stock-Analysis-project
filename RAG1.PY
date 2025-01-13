# from langchain_huggingface import HuggingFaceEndpoint
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def load_stock_data(csv_path):
    """
    Load stock data from CSV file into a pandas DataFrame
    """
    return pd.read_csv(csv_path)

def setup_llm():
    """
    Setup the HuggingFace LLM model
    """
    # Set your API token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    
    # Create the HuggingFaceEndpoint instance
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-large",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        task="text2text-generation",
        model_kwargs={
            "temperature": 0.5,
            "max_new_tokens": 150,
            "stop": ["\n"]
        }
    )
    return llm

def create_stock_agent(df, llm):
    """
    Create a LangChain agent that can analyze the DataFrame
    """
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True  # Only enable this in a safe environment
    )

def process_stock_query(agent, query):
    """
    Process a natural language query about stocks
    """
    try:
        response = agent.run(query)
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"

def main():
    try:
        # Load the stock data
        print("Loading stock data...")
        df = load_stock_data('sample.csv')
        
        # Setup the LLM
        print("Setting up the language model...")
        llm = setup_llm()
        
        # Create the agent
        print("Creating the analysis agent...")
        agent = create_stock_agent(df, llm)
        
        # Example queries from the assignment
        example_queries = [
            "What is the stock price of AAPL?",
            "What is the performance of TSLA?",
            "What is the P/E ratio of MSFT?",
            "Which stock has the highest price?",
            "What is the average stock price?",
            "Which stocks are in the Technology sector?",
            "What is the market cap of Amazon?"
        ]
        
        print("\nStock Analysis Program")
        print("=====================")
        
        while True:
            print("\nOptions:")
            print("1. Run example queries")
            print("2. Enter your own query")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                for query in example_queries:
                    print(f"\nQuery: {query}")
                    print("Answer:", process_stock_query(agent, query))
            elif choice == '2':
                query = input("\nEnter your query: ")
                print("Answer:", process_stock_query(agent, query))
            elif choice == '3':
                break
            else:
                print("Invalid choice. Please try again.")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
