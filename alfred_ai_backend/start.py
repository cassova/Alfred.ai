import argparse
import logging
from typing import Optional
from alfred_ai_backend.core.agent import AgentWrapper
from alfred_ai_backend.config import Config
from alfred_ai_backend.models.llama_cpp_local.mistral_instruct import MistralInstruct

# TODO: logger = logging.getLogger(__name__)

def configure_logger(debug_mode: bool, log_file: Optional[str]=None):
    """Configure the logging

    Args:
        debug_mode (bool): Enable debug logging
        log_file (Optional[str], optional): The output log file. Defaults to None.
    """
    config = Config()
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


def main():
    parser = argparse.ArgumentParser(
        description="Alfred.ai",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-t", "--task", type=str, help="The task to do")
    parser.add_argument("-l", "--log_file", type=str, help="The output log file")
    parser.add_argument("-d", "--debug", action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    configure_logger(args.debug, args.log_file)

    logging.info(f"Starting Alfred.ai")
    llm_wrapper = MistralInstruct()
    agent = AgentWrapper(llm_wrapper)

    if not args.task:
        print("Hello, give me a task to do...")
        try:
            while True:
                user_input = input(">>> ")
                resp = agent.start_task(user_input)
                print("Result: ", resp)
        except KeyboardInterrupt:
            print(" *** ctrl+c was pressed ***")

    logging.info(f"Shutting down")

if __name__ == "__main__":
    main()