import requests
from datetime import datetime, timedelta
from os import getcwd
import pandas as pd
from pathlib import Path

"""
Skapar data-set som används för att träna neuronnäten. Datan hämtas via cryptoexchange
Coinbase. 
"""

# Valutakoden för valutan som data-setet ska baseras på
currency_code = "ETH"
granularity = 5  # I minuter (1, 5, 10, 60 [1h], 360 [6h], 1440 [24h])

# Start och slutdatum för datan i formatet (år, månad, dag)
start_time, end_time = datetime(2018, 1, 1), datetime(2020, 12, 31)
init_time = start_time

Path(getcwd() + "/Datasets").mkdir(exist_ok=True)

day_buffer26 = []
while start_time < end_time:
    query = {
        "start": start_time.isoformat() + "Z",
        "end": (start_time + timedelta(days=1)).isoformat() + "Z",
        "granularity": str(granularity * 60),
    }
    """
    Begär data från coinbase API genom HTML requestar. Det finns en gräns för mängden data per request.
    Därför görs callsen tills datan täcker det specificerade tidsintervallet.
    """
    # time, low, high, open, close, volume
    (
        response := requests.get(
            "https://api.pro.coinbase.com/products/" + currency_code + "-EUR/candles",
            params=query,
        ).json()
    ).reverse()
    print(start_time.date())
    start_time += timedelta(days=1)
    for item in response:
        day_buffer26.append(item)

frame = pd.DataFrame(
    day_buffer26, columns=["time", "low", "high", "open", "close", "volume"]
)

# MACD divergence
macd = (
    pd.to_numeric(frame["close"]).ewm(span=6*288, adjust=False).mean()
    - pd.to_numeric(frame["close"]).ewm(span=13*288, adjust=False).mean()
)
macd_signal = macd.ewm(span=4.5*288, adjust=False).mean()

# Bollinger bands
sma = frame["close"].rolling(10*288).mean()
std = frame["close"].rolling(10*288).std()
bb_upper_dist = pd.Series.to_frame(
    (sma + std * 2) - frame["close"], name="bb_upper_dist"
)
bb_lower_dist = pd.Series.to_frame(
    frame["close"] - (sma - std * 2), name="bb_lower_dist"
)

frame = (
    pd.concat(
        [
            frame,
            pd.Series.to_frame(macd - macd_signal, name="macd"),
            bb_upper_dist,
            bb_lower_dist,
        ],
        axis=1,
    )
    .drop(pd.Series(range(0, 7488))).drop(['time', 'low', 'high', 'open'], axis=1)
    .to_csv(
        "Datasets/"
        + currency_code
        + "–"
        + str(init_time.date())
        + "–"
        + str(end_time.date())
        + ".csv",
        index=False,
    )
)
