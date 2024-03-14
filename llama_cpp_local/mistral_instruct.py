import json
from llama_cpp import Llama
from typing import Optional, Tuple
import logging

from llama_cpp_local.prompts import DEFAULT_SYSTEM_PROMPT, INSTRUCTION_FORMAT

MODEL_PATH = "D:/llama/models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<s> ", " </s>"

class MistralInstruct():
    def __init__(self):
        self._llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=-1, # Use GPU acceleration
            n_ctx=32700, # Set the context window
            # seed=1337, # Uncomment to set a specific seed
        )


    def _create_system_message(self, system_prompt: str) -> str:
        return f"{B_SYS} {B_INST} {system_prompt} {E_INST} "
    
    def _create_prompt(self, context: str, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        if not system_prompt:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        if context:
            return context + "\n" + INSTRUCTION_FORMAT.replace('{query}', user_prompt)
        return self._create_system_message(system_prompt) + "\n" + INSTRUCTION_FORMAT.replace('{query}', user_prompt)

    def _extact_output_json(self, response: str) -> dict:
        output_json = '{\n"tool_name": ' + response
        output_json = output_json.strip()
        if output_json.endswith("```"):
            output_json = output_json[:-3]
        return json.loads(output_json)


    def message(self, 
        user_prompt: str,
        context: Optional[str] = "",
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, str]:
        logging.debug(f"Checks: system_prompt: {True if system_prompt else False} | previous_dialog: {len(context)}")
        prompt = self._create_prompt(context, user_prompt, system_prompt)
        logging.debug(f"Prompt:\n{prompt}")
        
        resp = self._llm(
            prompt,
            max_tokens=2048,
            stop=["User:", "```", "Assistant:"],
            echo=False,
        )
        logging.debug(f"Response:\n{json.dumps(resp)}")
        response = resp['choices'][0]['text']
        context = prompt + response
        message = self._extact_output_json(response)
        logging.debug(f"Message:\n{json.dumps(message,indent=2)}")

        return message, context