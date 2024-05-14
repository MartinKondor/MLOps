"""
Simple OpenAI API wrapper that uses cache for repeated prompts.
For models, see: https://openai.com/api/pricing/
"""
from typing import List, Dict, Any

from openai import OpenAI as InternalOpenAI
import logging
import tiktoken


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Ask for the API key if its not defined
if "OPENAI_API_KEY" not in globals():
  OPENAI_API_KEY = input("OPENAI_API_KEY:")


def count_tokens(
    text: str,
    model_name: str = "gpt-3.5-turbo"
) -> int:
  token_encoder = tiktoken.encoding_for_model(model_name)
  return len(token_encoder.encode(text))


class OpenAIResponse:
  prompt: str
  message: Any
  no_input_tokens: int
  no_output_tokens: int

class OpenAI:
  
  # Static cache to store responses
  __cache: List[OpenAIResponse] = []

  def __init__(self):
    try:
      self.client = InternalOpenAI(
        api_key=OPENAI_API_KEY
      )
    except NameError as e:
      err_msg = "You should define OPENAI_API_KEY first!"
      logging.error(err_msg)
      raise Exception(err_msg)

  # Helper function to clean the text by converting to lowercase and stripping whitespace
  def normalize_text(self, text: str) -> str:
    return text.lower().strip()

  def __search_in_cache(self, prompt: str) -> str | None:
    for cache_obj in self.__cache:
      np1, np2 = self.normalize_text(cache_obj.prompt), self.normalize_text(prompt)
      if np1 == np2:
        logging.info("Response found in cache!")
        return cache_obj.message
    return None

  def __save_to_cache(self, prompt: str, message: Any) -> None:
    oairesp = OpenAIResponse()
    
    oairesp.prompt = prompt
    oairesp.no_input_tokens = count_tokens(prompt)
    
    oairesp.message = message
    oairesp.no_output_tokens = count_tokens(message)
    
    self.__cache.append(oairesp)

  def send_prompt(
      self,
      prompt: str,
      model: str = "gpt-3.5-turbo-0125",
      previous_messages: List[Dict[str, str]] = []
  ) -> str:
    cache_response = self.__search_in_cache(prompt)
    if cache_response is not None:
      return cache_response.content

    completion = self.client.chat.completions.create(
      model=model,
      messages=[
        *previous_messages,
        {"role": "user", "content": prompt}
      ]
    )
    message = completion.choices[0].message
    self.__save_to_cache(prompt, message)
    return message.content
