import sys
import logging


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            if len(line) > 2:  # Skip some superficial messaging like '.' progress indicators
                self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

class RedirectStdStreamsToLogger(object):
    def __init__(self, logger=None):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger('RedirectLogger')
        else:
            self.logger = logger

    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = StreamToLogger(self.logger, logging.INFO)
        sys.stderr = StreamToLogger(self.logger, logging.ERROR)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
