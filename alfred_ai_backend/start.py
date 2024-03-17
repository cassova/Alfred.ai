import argparse
import logging
from typing import Optional, Type
import importlib
from alfred_ai_backend.core.agent import AgentWrapper
from alfred_ai_backend.core.config import Config
from alfred_ai_backend.models.llm import LlmWrapper
from alfred_ai_backend.models.llama_cpp_local.mistral_instruct import MistralInstruct

logger = logging.getLogger(__name__)

def configure_logger(config: Config, debug_mode: bool, log_file: Optional[str]=None):
    """Configure the logging

    Args:
        config (Config): The loaded configuration
        debug_mode (bool): Enable debug logging
        log_file (Optional[str], optional): The output log file. Defaults to None.
    """
    logging.getLogger().handlers.clear()
    log_config = config.get("logging")

    handlers = []
    if log_config.get('verbose',False)==True:
        handlers.append(logging.StreamHandler())
    log_file = log_file if log_file else log_config['file']
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    if len(handlers)==0: return

    logging.basicConfig(
        level=getattr(logging, log_config['level']) if not debug_mode else logging.DEBUG,
        format=log_config.get('format','%(asctime)s - %(levelname)s - %(message)s'),
        handlers=handlers,
    )

def get_model_class(config: Config, model: Optional[str] = 'default_model') -> LlmWrapper:
    """This dyanmically loads the LLM model to be used by the agent

    Args:
        config (Config): The loaded configuration
        model (Optional[str], optional): The model load. Defaults to 'default_model'.

    Raises:
        Exception: If the configuration is missing a definition for all models
        Exception: if the configuration is missing the specified model's definition

    Returns:
        Type[LlmWrapper]: A subclass of LlmWrapper
    """
    all_models_config = config.get('models')
    if all_models_config == None:
        error = "Missing configuration for models"
        logger.error(error)
        raise Exception(error)
    if model not in all_models_config:
        error = f"Unable to activate {model} because it's not defined in the config"
        logger.error(error)
        raise Exception(error)
    
    class_name = all_models_config[model].get('name')
    module_name = all_models_config[model].get('module')
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        logger.error(f"Failed to import module_name '{module_name}'")
        raise

    logger.info(f"Loaded model {module_name}")
    return getattr(module, class_name)()

def main():
    parser = argparse.ArgumentParser(
        description="Alfred.ai",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-t", "--task", type=str, help="The task to do")
    parser.add_argument("-l", "--log_file", type=str, help="The output log file")
    parser.add_argument("-d", "--debug", action='store_true', help='Enable debug logging')
    parser.add_argument("-m", "--model", type=str, help="The model to use", default='default_model')
    args = parser.parse_args()

    config = Config()
    configure_logger(config, args.debug, args.log_file)

    logger.info(f"Starting Alfred.ai")
    llm_wrapper = get_model_class(config, args.model)
    agent = AgentWrapper(config, llm_wrapper)

    if not args.task:
        print("Hello, give me a task to do...")
        try:
            while True:
                user_input = input(">>> ")

                if len(user_input)<8 and (user_input.lower().startswith('exit') or user_input.lower().startswith('quit')):
                    break

                if len(user_input)>0:
                    resp = agent.start_task(user_input)
                    logger.info(f"Response: {resp}")
                    print(resp.get('output', ' [[no response]]'))
        except KeyboardInterrupt:
            print(" *** ctrl+c was pressed ***")

    logger.info(f"Shutting down")

if __name__ == "__main__":
    main()
