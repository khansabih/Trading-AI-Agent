import time
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from newsapi import NewsApiClient
from textblob import TextBlob
from datetime import datetime
from ta.momentum import RSIIndicator

NEWS_API_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
news_api = NewsApiClient(api_key=NEWS_API_KEY)

indian_nifty100_tickers = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "LT", "ITC", "AXISBANK",
    "KOTAKBANK", "SBIN", "BHARTIARTL", "ASIANPAINT", "DMART", "SUNPHARMA", "BAJFINANCE",
    "HCLTECH", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "ADANITRANS",
    "BAJAJ-AUTO", "BAJAJFINSV", "BPCL", "CIPLA"
]

stock_data = []

for symbol in indian_nifty100_tickers:
    ns_ticker = symbol + ".NS"
    bo_ticker = symbol + ".BO"
    
    for ticker in [ns_ticker, bo_ticker]:
        try:
            print(f"Trying {ticker}...")
            data = yf.Ticker(ticker).history(period="5d")
            data = data.dropna(subset=["Close"])
            if len(data) < 2:
                print(f"Not enough data for {ticker}, skipping.")
                continue

            latest = data.iloc[-1]
            previous = data.iloc[-2]

            change = (latest["Close"] - previous["Close"]) / previous["Close"]
            stock_data.append({
                'Ticker': ticker,
                'Latest Close': round(latest["Close"], 2),
                'Previous Close': round(previous["Close"], 2),
                'Change (%)': round(change, 4),
                'Volume': int(latest["Volume"])
            })
            break  # success, no need to try fallback
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            continue

df_market = pd.DataFrame(stock_data)
print("\n--- Indian Market Snapshot ---")
print(df_market.sort_values(by='Change (%)', ascending=False))

# SENTIMENT ANALYSIS OF ALL THE TICKERS
sentiment_data = []

for ticker in indian_nifty100_tickers:
    try:
        print(f"\nFetching news for {ticker}...")
        response = news_api.get_everything(
            q=ticker,
            language='en',
            sort_by='relevancy',
            page_size=5
        )

        articles = response.get('articles', [])
        sentiments = []

        for article in articles:
            title = article.get('title', '')
            desc = article.get('description', '')
            content = f"{title}. {desc}".strip()

            if content:
                blob = TextBlob(content)
                sentiments.append(blob.sentiment.polarity)

        if sentiments:
            avg_sentiment = round(sum(sentiments) / len(sentiments), 3)
            print(f"Sentiment: {avg_sentiment:.3f} based on {len(sentiments)} articles.")
        else:
            avg_sentiment = None
            print("No valid articles for sentiment analysis.")

        sentiment_data.append({
            'Ticker': ticker,
            'Sentiment Score': avg_sentiment,
            'Articles Found': len(articles)
        })

        time.sleep(1)

    except Exception as e:
        print(f"Error with {ticker}: {e}")
        sentiment_data.append({
            'Ticker': ticker,
            'Sentiment Score': None,
            'Articles Found': 0
        })

df_sentiment = pd.DataFrame(sentiment_data)
df_sentiment = df_sentiment.sort_values(by='Sentiment Score', ascending=False)
print("\n--- Sentiment Snapshot ---")
print(df_sentiment)

# Remove '.NS' from market ticker names for consistency
df_market['Ticker'] = df_market['Ticker'].str.replace('.NS', '', regex=False)
merged_df = df_market.merge(df_sentiment, on='Ticker', how='left')
print("\n--- Combined Market + Sentiment Snapshot ---")
print(merged_df.sort_values(by='Change (%)', ascending=False))

# Filter out tickers with no sentiment score
filtered_df = merged_df[merged_df['Sentiment Score'].notnull()]
threshold = 0.005
filtered_df = filtered_df[filtered_df['Change (%)'].abs() > threshold]

print("\n--- Filtered Snapshot ---")
print(filtered_df.sort_values(by='Change (%)', ascending=False))

# Bullish & Bearish
bullish_stocks = filtered_df[
    (filtered_df['Sentiment Score'] > 0.05) & (filtered_df['Change (%)'] > 0)
]
print("\n--- Bullish Candidates ---")
print(bullish_stocks)

bearish_stocks = filtered_df[
    (filtered_df['Sentiment Score'] < -0.05) & (filtered_df['Change (%)'] < 0)
]
print("\n--- Bearish Candidates ---")
print(bearish_stocks)

filtered_df['Composite Score'] = (
    filtered_df['Sentiment Score'] * 0.7 + filtered_df['Change (%)'] * 0.3
)
ranked_df = filtered_df.sort_values(by='Composite Score', ascending=False)

print("\n--- Ranked Recommendation ---")
print(ranked_df[['Ticker', 'Sentiment Score', 'Change (%)', 'Composite Score']].head(10))

# Trade Signals
buy_signals = filtered_df[
    (filtered_df['Sentiment Score'] > 0.05) & (filtered_df['Change (%)'] > 0.005)
]
sell_signals = filtered_df[
    (filtered_df['Sentiment Score'] < -0.05) & (filtered_df['Change (%)'] < -0.005)
]

print("\n--- Trade Signals ---")
print("📈 Buy Signals:")
print(buy_signals[['Ticker', 'Change (%)', 'Sentiment Score']])
print("\n📉 Sell Signals:")
print(sell_signals[['Ticker', 'Change (%)', 'Sentiment Score']])

# Save snapshot
date_str = datetime.now().strftime('%Y-%m-%d')
filtered_df.to_csv(f"market_sentiment_snapshot_{date_str}.csv", index=False)
buy_signals.to_csv(f"buy_signals_{date_str}.csv", index=False)


# Momentum Decay Analysis
def price_decay_momentum_analysis(ticker):
    try:
        print(f"\n📉 Analyzing price momentum decay for {ticker}...")
        data = yf.Ticker(ticker).history(period="90d")

        if len(data) < 10:
            print(f"⚠️ Not enough data for {ticker}")
            return

        data["Daily Return"] = data["Close"].pct_change()
        data["5D Rolling Return"] = data["Daily Return"].rolling(window=5).mean()

        rolling = data["5D Rolling Return"].dropna()
        peak_day = rolling.idxmax()
        peak_val = rolling.max()

        try:
            decay_day = rolling[peak_day:].loc[rolling[peak_day:] < 0.1 * peak_val].index[0]
            decay_days = (decay_day - peak_day).days
        except IndexError:
            decay_days = "Not decayed yet"

        print(f"🔍 Momentum peak at {peak_day.date()} with {peak_val:.4f} return")
        print(f"⏳ Momentum decay: {decay_days} days")

        # plt.figure(figsize=(10, 5))
        # plt.plot(rolling.index, rolling.values, label='5D Rolling Return')
        # plt.axvline(peak_day, color='green', linestyle='--', label='Peak')
        # if isinstance(decay_days, int):
        #     plt.axvline(decay_day, color='red', linestyle='--', label='Decay Point')
        # plt.title(f"{ticker} Momentum Decay Analysis")
        # plt.xlabel("Date")
        # plt.ylabel("5D Return")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

    except Exception as e:
        print(f"⚠️ Failed to analyze momentum for {ticker}: {e}")

# RSI indication
def compute_RSI(base_ticker, period=14):
    for suffix in [".NS", ".BO"]:
        try:
            ticker = base_ticker + suffix
            hist = yf.Ticker(ticker).history(period=f"{period + 1}d")
            hist = hist.dropna()
            if len(hist) < period:
                continue
            rsi = RSIIndicator(close=hist['Close'], window=period).rsi()
            latest_rsi = rsi.iloc[-1]
            return round(latest_rsi, 2)
        except Exception as e:
            print(f"Error calculating RSI for {ticker}: {e}")
            continue
    return None

# TESTING RSI FUNCTIONALITY
# tick = 'SUNPHARMA'
# rsi = compute_RSI(tick)
# print(f"RSI for {tick}: {rsi}")
# if rsi is not None and rsi < 30:
#     print(f"✅ {tick} has RSI {rsi} (<30) → Confident BUY signal")

# Run decay analysis + RSI + MACD + Volume Spike for each Buy Signal
for i, row in buy_signals.iterrows():
    price_decay_momentum_analysis(row['Ticker'] + ".NS")
    rsi = compute_RSI(row['Ticker'])
    print(f"RSI for {row['Ticker']}: {rsi}")
    if rsi is not None and rsi < 30:
        print(f"✅ {row['Ticker']} has RSI {rsi} (<30) → Confident BUY signal")
