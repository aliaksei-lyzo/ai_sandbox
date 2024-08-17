from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.json import JsonOutputParser

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None:
  raise ValueError("No Open AI Api-key provided")

translation_model = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.5)

json_output_parser = JsonOutputParser()

openai_translation_chain = translation_model | json_output_parser