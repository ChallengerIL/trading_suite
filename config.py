from datetime import time
from datetime import datetime, timedelta
from credentials import TESTING_LOGIN, TESTING_PASSWORD
import pytz
import math

# TIMEZONE = pytz.timezone('Asia/Tel_Aviv')
TIMEZONE = pytz.timezone('Europe/London')

# tp, sl, rsi period, mfi period, ema_7 period, ema_20 period, wr_21 period, rsi level, mfi level, wr_21 level

# Correlation
# ['GBPUSD', 'EURUSD', 'AUDUSD']
# ['USDJPY', 'USDCAD', 'USDCHF', 'EURJPY']

# Define the current trend with HMA 12 relative to HMA 80

# If TSV is UP and HMA 12 crosses HMA 80 to the top - Buy, sell if it's mirrored

STRATEGIES = {
    "example_strategy": {
        "tfs": ["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
        "pairs": ['GBPUSD'],
        "indicators": {
            "macd": {
                "M1": {
                    "fast_ema": [12],
                    "slow_ema": [26],
                    "signal_ema": [9],
                    "plotting": {
                        "ax": [1],
                    },
                },
            },
            # "stochastic": {
            #     "M1": {
            #         "p": [14],
            #         "sma_p": [3],
            #         "plotting": {
            #             "color": [["white", "green"]],
            #             "ax": [1],
            #             "levels": [[80, 20]],
            #             "ymin": 0,
            #             "ymax": 100,
            #         },
            #     },
            # },
            # "mfi": {
            #     "M1": {
            #         "p": [14],
            #         "plotting": {
            #             "color": ["white", "red"],
            #             "ax": [1],
            #             "levels": [[80, 20]],
            #             "ymin": 0,
            #             "ymax": 100,
            #         },
            #     },
            # },
            # "mml": {
            #     "H1": {
            #         "window": [64],
            #         "plotting": {
            #             "color": [["blue", "blue", "blue", "blue", "red", "blue", "blue", "blue", "blue"]],
            #         },
            #     },
            # },
            # "supertrend": {
            #     "H1": {
            #         "p": [12],
            #         "multiplier": [3],
            #         "plotting": {
            #             "color": [["blue", "green", "red"]],
            #         },
            #     },
            # },
            # "rsi": {
            #     "H1": {
            #         "p": [14],
            #         "plotting": {
            #             "color": ["yellow"],
            #             "ax": [2],
            #             "levels": [[20, 80]],
            #             "ymin": 0,
            #             "ymax": 100,
            #         },
            #     },
            # },
            # "wr": {
            #     "M1": {
            #         "p": [21],
            #         "plotting": {
            #             "color": ["red"],
            #             "ax": [3],
            #             "levels": [[-20, -80]],
            #             "ymin": -100,
            #             "ymax": 0,
            #         },
            #     },
            # },
            # "tsv": {
            #     "M1": {
            #         "p": [13],
            #         "ma_p": [7],
            #         "plotting": {
            #             "ax": [1],
            #         },
            #     },
            # },
            # "ichimoku_cloud": {
            #     "H1": {
            #         "plotting": {
            #             "color": [["green", "yellow"]],
            #         },
            #     },
            # },
            # "wma": {
            #     "H1": {
            #         "p": [12, 80],
            #         "plotting": {
            #             "color": ["white", "blue"],
            #         },
            #     },
            # },
            # "dema": {
            #     "H1": {
            #         "p": [12, 80],
            #         "plotting": {
            #             "color": ["white", "blue"],
            #         },
            #     },
            # },
            # "tema": {
            #     "H1": {
            #         "p": [12, 80],
            #         "plotting": {
            #             "color": ["white", "blue"],
            #         },
            #     },
            # },
            # "ema": {
            #     "H1": {
            #         "p": [12, 80],
            #         "plotting": {
            #             "color": ["white", "blue"],
            #         },
            #     },
            # },
            # "hma": {
            #     "H1": {
            #         "p": [12, 80],
            #         "plotting": {
            #             "color": ["green", "red"],
            #         },
            #     },
            # },
            # "tma": {
            #     "M1": {
            #         "p": [20],
            #         "atr_p": [100],
            #         "multiplier": [2],
            #         "plotting": {
            #             "color": [["white", "white", "white"]],
            #         },
            #     },
            #     "M15": {
            #         "p": [20],
            #         "atr_p": [100],
            #         "multiplier": [2],
            #         "plotting": {
            #             "color": [["red", "red", "red"]],
            #         },
            #     },
            #     "H1": {
            #         "p": [20],
            #         "atr_p": [100],
            #         "multiplier": [2],
            #         "plotting": {
            #             "color": [["blue", "blue", "blue"]],
            #         },
            #     },
            # },
        },
        "bounds_low": [20, 20, -20, 1, 10, 5, 5, 10],
        "bounds_high": [300, 100, -1, 20, 100, 30, 15, 30],
        "tp": 1000,
        "sl": -20,
        "total_trades": 1000,
        "update_weekday": 5,
        "update_time": time(0, 0, 0, tzinfo=TIMEZONE),
        "update_period": 25,
        "plotting_rows": 2,
        "training_period": 500,
    },
}

# Hunter
# "bounds_low": [10, -15, 10, 1, -1, -1, -1, -1, -1, -1, -1, 10, -5, -5, -1, 1, 0, 5, 0, 15, -1, -1, -1, -1,
#                -1, -1],
# "bounds_high": [50, -5, 100, 10, 1, 1, 1, 1, 1, 1, 1, 30, 5, 5, 1, 30, 1, 20, 10, 25, 1, 1, 1, 1, 1, 1],

# os.system("python ga.py " + "GBPUSD " + str(START) + " " + str(END) + " test.py")

# END = datetime(2022, 1, 1, tzinfo=TIMEZONE)
# START = datetime(2021, 1, 1, tzinfo=TIMEZONE)

# # Training Period
END = datetime.now().replace(tzinfo=TIMEZONE)
# START = END - timedelta(days=STRATEGIES["hunter"]["training_period"])

# END = datetime.now().replace(tzinfo=TIMEZONE)
# START = datetime(2022, 1, 1, tzinfo=TIMEZONE)

# END = datetime(2022, 1, 1, tzinfo=TIMEZONE)
START = END - timedelta(days=STRATEGIES["example_strategy"]["training_period"])


def current_time():
    return datetime.now().replace(tzinfo=TIMEZONE)


# MetaTrader 5
# TERMINAL_PATH = 'C:\\Program Files\\Alpari MT5\\terminal64.exe'
# LOGIN = 0000000
# SERVER = "Alpari-MT5-Demo"
# PASSWORD = password
FILES_DIR = "files/"

# For Testing
TERMINAL_PATH = 'C:\\Program Files\\Alpari MT5 2\\terminal64.exe'
LOGIN = TESTING_LOGIN
SERVER = "Alpari-MT5-Demo"
PASSWORD = TESTING_PASSWORD


# Trading
START_TRADING = 7
END_TRADING = 21
DEVIATION = 20
BUFFER_BARS = 360
SYMBOL_HEAD = ""
SYMBOL_TAIL = "_i"
PAIRS = ['GBPUSD', 'AUDUSD', 'USDCAD', 'USDJPY', 'EURJPY']

# GA Settings
# TP, SL, ENTRY LONG, ENTRY SHORT, RSI VWAP PERIOD
PIPS_MULTIPLIER = 0.005
# PIPS_MULTIPLIER = 0.1
CUSHION = 0

# Try disabling balance tracking
# Play with the crowding factor
# Add drawback parameter to the fitness function
# POPULATION_SIZE = 300
POPULATION_SIZE = math.ceil(len(STRATEGIES["example_strategy"]["bounds_high"]) / 10) * 100
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.6
MAX_GENERATIONS = 20
HALL_OF_FAME_SIZE = int(POPULATION_SIZE / 10)
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation

# 6 1180.2
# POPULATION_SIZE = 300
# P_CROSSOVER = 0.9  # probability for crossover
# P_MUTATION = 0.6
# MAX_GENERATIONS = 50
# HALL_OF_FAME_SIZE = int(POPULATION_SIZE / 10)
# CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation

# Other Settings
SLEEP_DAY = 5
WEEKEND_SLEEP = True
