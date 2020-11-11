import colorlog
import logging

formatter = colorlog.ColoredFormatter(
    fmt='[%(log_color)s%(levelname)s%(reset)s] %(message)s (%(asctime)s)',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red, bold'
    }
)

streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)

logger = logging.getLogger('U-Net')
logger.addHandler(streamHandler)
logger.setLevel(logging.INFO)