from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def get_ohlcv_data(ticker: str, start_date: str, end_date: str) -> dict:
    """
    지정된 ticker와 날짜 범위에 대해 OHLCV 가격 데이터를 시뮬레이션하여 반환.

    Args:
        ticker (str): 종목 티커
        start_date (str): 시작일 (YYYY-MM-DD)
        end_date (str): 종료일 (YYYY-MM-DD)

    Returns:
        dict: {'ticker': str, 'price_history': List[float], 'market_cap': int, 'start_date': str, 'end_date': str}
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    num_days = (end - start).days + 1

    price_history = np.cumprod(1 + np.random.normal(0.0005, 0.01, num_days)) * 100
    market_cap = int(np.random.uniform(1e10, 5e11))  # 1조 ~ 50조

    return {
        'ticker': ticker,
        'price_history': price_history.tolist(),
        'market_cap': market_cap,
        'start_date': start_date,
        'end_date': end_date
    }
