from langchain_core.messages import HumanMessage, SystemMessage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from openai_model import openai_translation_chain

app = FastAPI()

class TranslationRequest(BaseModel):
  target_languages: List[str]
  document: str

@app.post("/articles/translation")
def translate_article(translation_request: TranslationRequest):
    target_languages, document = translation_request
    messages = [
      SystemMessage(content=f"""
                    You are a translator, that translates to {str(target_languages)}.
                    Response should be a JSON string with fields corresponding to the language ISO 639-2 code
                    and the values should be the corresponding translation.
                    If the actual document language could not be identified return
                    JSON with field "error" and value "Couldn't identify document language"
                    """),
      HumanMessage(content=str(document))
    ]
    model_data = openai_translation_chain.invoke(messages)
    if "error" in model_data:
       raise HTTPException(status_code=400, detail=model_data["error"])
    return model_data
