from sampling.sampling import minute_to_toronto_session_daily_ohlc
from pathlib import Path
import pandas as pd
from datetime import date


def _make_rows(d, times_prices):
    # times_prices: list of (HH:MM, o,h,l,c,vol)
    rows = []
    for t, o, h, l, c, v in times_prices:
        rows.append([d.strftime("%Y.%m.%d"), t, o, h, l, c, v])
    return pd.DataFrame(rows, columns=["date", "time", "open", "high", "low", "close", "volume"])


def test_toronto_session_and_dst_shift():
    # Winter date: Jan 4, 2010 (Toronto is EST)
    winter = _make_rows(
        date(2010, 1, 4),
        [
            ("09:29", 1.0, 1.0, 1.0, 1.0, 1),
            ("09:30", 2.0, 2.1, 1.9, 2.0, 1),
            ("16:00", 3.0, 3.2, 2.8, 3.1, 1),
            ("16:01", 9.0, 9.0, 9.0, 9.0, 1),
        ],
    )

    # DST date: Mar 15, 2010 (Toronto is EDT).
    # To capture 09:30 Toronto, the fixed-EST timestamp must be 08:30.
    dst = _make_rows(
        date(2010, 3, 15),
        [
            ("08:29", 10.0, 10.0, 10.0, 10.0, 1),
            ("08:30", 11.0, 11.5, 10.5, 11.2, 2),  # should be included as 09:30 Toronto
            ("15:00", 12.0, 13.0, 11.0, 12.5, 3),  # should be included as 16:00 Toronto
            ("15:01", 99.0, 99.0, 99.0, 99.0, 1),
        ],
    )

    df = pd.concat([winter, dst], ignore_index=True)

    daily = minute_to_toronto_session_daily_ohlc(df)

    # Winter day checks (09:30..16:00 include exactly 2 bars: 09:30 and 16:00)
    d_w = pd.Timestamp("2010-01-04")
    assert d_w in daily.index
    assert daily.loc[d_w, "n_bars"] == 2
    assert daily.loc[d_w, "open"] == 2.0
    assert daily.loc[d_w, "high"] == max(2.1, 3.2)
    assert daily.loc[d_w, "low"] == min(1.9, 2.8)
    assert daily.loc[d_w, "close"] == 3.1
    assert daily.loc[d_w, "volume"] == 2  # 1 + 1

    # DST day checks: should include 08:30 and 15:00 fixed-EST (which become 09:30 and 16:00 Toronto)
    d_s = pd.Timestamp("2010-03-15")
    assert d_s in daily.index
    assert daily.loc[d_s, "n_bars"] == 2
    assert daily.loc[d_s, "open"] == 11.0
    assert daily.loc[d_s, "high"] == max(11.5, 13.0)
    assert daily.loc[d_s, "low"] == min(10.5, 11.0)
    assert daily.loc[d_s, "close"] == 12.5
    assert daily.loc[d_s, "volume"] == 5  # 2 + 3


# Run ad-hoc if you want:
if __name__ == "__main__":
    test_toronto_session_and_dst_shift()
    print("OK")