import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = "your_api_key"

def invokeLLM(input_variety='Fortuna'):
    """
    This functions employs OpenAI's GPT 3.5 LLM for returning the response of user queries (pre-defined).
    It uses API calls and integrates the LLM with LangChain for interfacing.

    Parameters:
        input_variety: The strawberry variety returned by the classification model after inference.
    
    Returns:
        None
    """
    print("Initializing OpenAI's GPT-3.5 Turbo model.")
    # Initialize the OpenAI LLM
    llm = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0)

    # First step in chain
    first_template = """Question: Give me a general overview of the {variety} strawberry.
    Display the title, 'General Overview' at the start of the paragraph. 
    Restrict your response to a single paragraph consisting of 50 words."""

    first_prompt = PromptTemplate(
        input_variables=["variety"],
        template=first_template
    )
    chain_one = LLMChain(llm=llm, prompt=first_prompt)

    # Second step in chain
    second_template = """Question: What are the unique characteristics of the {variety} strawberry.
    Display the title, 'Unique Characteristics' at the start of the paragraph. 
    Restrict your response to a single paragraph consisting of 50 words."""

    second_prompt = PromptTemplate(
        input_variables=["variety"],
        template=second_template
    )
    chain_two = LLMChain(llm=llm, prompt=second_prompt)
    
    # Third step in chain
    third_template = """Question: In what conditions does the {variety} strawberry grow?
    Display the title, 'Growing Conditions' at the start of the paragraph. 
    Restrict your response to a single paragraph consisting of 50 words."""

    third_prompt = PromptTemplate(
        input_variables=["variety"],
        template=third_template
    )
    chain_three = LLMChain(llm=llm, prompt=third_prompt)

    # Fourth step in chain
    fourth_template = """Question: Articulate a single food recipe for the {variety} strawberry.
    Display its title at the start.
    Detail each step including metrics such as ingredients, quantity, time, etc.
    Use bullets to formulate your response.
    Restrict your response to 100 words."""

    fourth_prompt = PromptTemplate(
        input_variables=["variety"],
        template=fourth_template
    )
    chain_four= LLMChain(llm=llm, prompt=fourth_prompt)

    # Combine the two chains together
    overall_chain = SimpleSequentialChain(chains=[chain_one, chain_two, chain_three, chain_four], verbose=True)

    try:    
        overall_chain.invoke(input_variety)
    except Exception as e:
        print(f"An error occurred: {e}") 