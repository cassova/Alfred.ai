import argparse
import logging
from typing import Optional
#from openai_api.chat import message
from llama_cpp_local.mistral_instruct import MistralInstruct


def configure_logger(quiet_mode: bool, debug_mode: bool, log_file: Optional[str]=None):
    """Configure the logging

    Args:
        quiet_mode (bool): Suppress console output
        debug_mode (bool): Enable debug logging
        log_file (Optional[str], optional): The output log file. Defaults to None.
    """
    logging.getLogger().handlers.clear()

    handlers = []
    if not quiet_mode:
        handlers.append(logging.StreamHandler())
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    if len(handlers)==0: return

    logging.basicConfig(
        level=logging.INFO if not debug_mode else logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Alfred.ai",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-t", "--task", type=str, help="The task to do")
    parser.add_argument("-l", "--log_file", type=str, default="output.log", help="The output log file")
    parser.add_argument("-d", "--debug", action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    #configure_logger(True, args.debug, args.log_file)
    configure_logger(True, True, args.log_file)  # Hardcode enable debug mode

    logging.info(f"Starting Alfred.ai")
    chat = MistralInstruct()

    if not args.task:
        print("Hello, give me a task to do...")
        try:
            context = ""
            while True:
                user_input = input(">>> ")
                message, context = chat.message(user_input, context=context)
                if message["tool_name"] == "Final Answer":
                    print(message["input"])
        except KeyboardInterrupt:
            print(" *** ctrl+c was pressed ***")

    logging.info(f"Shutting down")

if __name__ == "__main__":
    main()
