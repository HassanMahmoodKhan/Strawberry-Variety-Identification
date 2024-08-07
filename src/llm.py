import os
import warnings
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module='langchain_core._api.deprecation')

# Load environment variables from the .env file
load_dotenv()

# Access the API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is loaded successfully
if openai_api_key is None:
    raise ValueError("API key not found. Please set OPENAI_API_KEY in your .env file.")

print("Initializing OpenAI's GPT-3.5 Turbo model.")
# Initialize the OpenAI LLM
llm = OpenAI(api_key=openai_api_key, model_name='gpt-3.5-turbo-instruct', temperature=0)

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
chain_four = LLMChain(llm=llm, prompt=fourth_prompt)

# Combine the four chains together
overall_chain = SimpleSequentialChain(chains=[chain_one, chain_two, chain_three, chain_four],
                                    verbose=True)