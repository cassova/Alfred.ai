import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed
import logging
import json
from typing import Optional, List

DEFAULT_MODEL = "gpt-4-0125-preview"
DEFAULT_TEMPERATURE = 0.0

#@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def message(
    user_prompt: str,
    previous_dialog: Optional[list] = [],
    system_prompt: Optional[str] = None,
    functions: Optional[list] = None,
    function_call: Optional[dict] = None,
    temperature: Optional[float] = 0.0,
    model: Optional[str] = None
) -> List[dict]:
    client = OpenAI(api_key=os.getenv('openai_api_key'))

    messages = previous_dialog.copy()
    messages.append({"role":"user", "content":user_prompt})
    logging.debug(f"Checks: system_prompt: {True if system_prompt else False} | previous_dialog: {len(previous_dialog)}")
    if system_prompt and len(previous_dialog) == 0:
        messages.insert(0, {"role":"system", "content":system_prompt})

    try:
        response = client.chat.completions.create(
            model=model if model else DEFAULT_MODEL,
            messages=messages,
            functions=functions,
            function_call=function_call,
            temperature=temperature if temperature else DEFAULT_TEMPERATURE,
        )
    except Exception as e:
        logging.warn(f"Failed to get OpenAI chat response with error: {e}")
        raise

    if function_call:
        messages.append({"role":"assistant", "content":response.choices[0].function_call.arguments})
    else:
        messages.append({"role":"assistant", "content":response.choices[0].message.content})

        
    logging.debug(f"Messages:\n{json.dumps(messages,indent=2)}")
    return messages
