import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import torch

def load_csv(file_path: str) -> pd.DataFrame:
    """Load stock data from CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def initialize_huggingface_model(model_name: str = "bigscience/bloom-560m"):
    """Initialize a Hugging Face text-generation model for LangChain."""
    # Initialize tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Create pipeline without explicit device mapping
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
    
    return HuggingFacePipeline(pipeline=pipe)

def create_chain(llm):
    """Create a processing chain using the modern LangChain syntax."""
    prompt = PromptTemplate.from_template(
        """Analyze the following stock market data and answer the question.
        
        Stock Data:
        {data}
        
        Question: {question}
        
        Answer: """
    )
    
    # Create the chain using the modern pipe syntax
    chain = (
        {"question": RunnablePassthrough(), "data": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def process_query(chain, df: pd.DataFrame, question: str) -> str:
    """Process a single query using the chain."""
    # Convert DataFrame to a more readable string format
    data_context = df.to_string(index=False, justify='left')
    try:
        return chain.invoke({"question": question, "data": data_context})
    except Exception as e:
        return f"Error processing query: {str(e)}"

def main():
    # File Path
    file_path = "sample.csv"
    
    # Load data
    print("Loading data...")
    stock_data = load_csv(file_path)
    
    # Initialize model
    print("Initializing model...")
    llm_model = initialize_huggingface_model()
    
    # Create chain
    print("Creating processing chain...")
    chain = create_chain(llm_model)
    
    # Example questions
    questions = [
        "What is the stock price of AAPL?",
        "What is the performance of MSFT?",
        "Which stock has the highest market cap?",
        "What is the sector of AMZN?"
    ]
    
    # Process questions
    print("\nProcessing queries...")
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = process_query(chain, stock_data, question)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()