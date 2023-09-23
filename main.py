from bot import Bot
from config import STRATEGIES

trading_bot = Bot(STRATEGIES["hunter"])

if __name__ == "__main__":
    trading_bot.run()
