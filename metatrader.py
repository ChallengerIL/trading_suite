import MetaTrader5 as mt5
from config import TERMINAL_PATH, LOGIN, SERVER, PASSWORD


def connect(logger=None):
    if mt5.initialize(TERMINAL_PATH, login=LOGIN, server=SERVER, password=PASSWORD):
        if logger is not None:
            logger.info("Established connection to MetaTrader 5")
    else:
        logger.info("Failed to establish connection to MetaTrader 5")
        mt5.shutdown()


def shutdown(logger=None):
    if logger is not None:
        logger.info("Connection to MetaTrader 5 has been shut down")
    mt5.shutdown()
