from bot import Bot
from config import STRATEGIES

trading_bot = Bot(STRATEGIES["example_strategy"])

if __name__ == "__main__":
    trading_bot.run()
