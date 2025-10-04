# Streamlit AI Trader - single-file app
# Features:
# - Download OHLCV using yfinance
# - Compute indicators (SMA, EMA, RSI, ATR, returns)
# - Train an XGBoost binary signal (forward return > threshold)
# - Simple backtest (next-bar execution) with slippage
# - Interactive charts and trade table
# - Upload CSV support
#
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import io
import joblib
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
import plotly.graph_objects as go
import plotly.express as px

# --- Helpers ---
@st.cache_data
def download_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
    if df.empty:
        return df
    df = df.reset_index()
    df.rename(columns={"Datetime":"datetime","Date":"datetime"}, inplace=True)
    df['datetime'] = pd.to_datetime(df['Date'] if 'Date' in df.columns else df['Datetime'])
    df.set_index('datetime', inplace=True)
    return df[['Open','High','Low','Close','Adj Close','Volume']].rename(columns={'Adj Close':'Adj_Close'})


def compute_indicators(df):
    df = df.copy()
    df['return_1'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()
    df.dropna(inplace=True)
    return df


def make_target(df, forward_bars=1, threshold=0.004):
    # target = 1 if forward return over forward_bars > threshold
    df = df.copy()
    df['future_close'] = df['Close'].shift(-forward_bars)
    df['fwd_return'] = (df['future_close'] - df['Close']) / df['Close']
    df['target'] = (df['fwd_return'] > threshold).astype(int)
    df.dropna(inplace=True)
    return df


def train_xgb(df, features, seed=42):
    X = df[features]
    y = df['target']
    tscv = TimeSeriesSplit(n_splits=5)
    metrics = {'accuracy':[], 'precision':[], 'recall':[], 'auc':[]}
    models = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = XGBClassifier(n_estimators=200, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=seed)
        model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_val,y_val)], verbose=False)
        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)[:,1]
        metrics['accuracy'].append(accuracy_score(y_val, preds))
        metrics['precision'].append(precision_score(y_val, preds, zero_division=0))
        metrics['recall'].append(recall_score(y_val, preds, zero_division=0))
        try:
            metrics['auc'].append(roc_auc_score(y_val, proba))
        except Exception:
            metrics['auc'].append(np.nan)
        models.append(model)
    # choose last model as final (or retrain on all data)
    final_model = XGBClassifier(n_estimators=200, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=seed)
    final_model.fit(X, y, verbose=False)
    return final_model, metrics


def backtest(df, model, features, forward_bars=1, slippage_pct=0.0005):
    # simple next-bar execution: enter at next open, exit after holding forward_bars bars (target logic used)
    data = df.copy()
    X = data[features]
    data['pred_proba'] = model.predict_proba(X)[:,1]
    data['signal'] = (data['pred_proba'] > 0.5).astype(int)
    trades = []
    capital = 100000.0
    equity = capital
    eq_curve = []
    position = 0
    for idx in range(len(data)-forward_bars):
        row = data.iloc[idx]
        if row['signal'] == 1:
            entry_price = data['Open'].iloc[idx+1]  # enter next bar open
            exit_price = data['Close'].iloc[idx+forward_bars]  # exit at close after holding
            if np.isnan(entry_price) or np.isnan(exit_price):
                eq_curve.append(equity)
                continue
            # slippage applied to entry and exit
            entry_price = entry_price * (1 + slippage_pct)
            exit_price = exit_price * (1 - slippage_pct)
            returns = (exit_price - entry_price) / entry_price
            pnl = equity * 0.01 * returns  # fixed 1% of capital per trade
            equity += pnl
            trades.append({
                'entry_time': data.index[idx+1],
                'exit_time': data.index[idx+forward_bars],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'returns': returns,
                'pnl': pnl
            })
        eq_curve.append(equity)
    # align eq_curve index
    eq_idx = data.index[:len(eq_curve)]
    eq_series = pd.Series(eq_curve, index=eq_idx)
    trades_df = pd.DataFrame(trades)
    summary = {
        'starting_capital': capital,
        'ending_capital': equity,
        'total_trades': len(trades_df),
        'net_profit': equity - capital,
        'win_rate': (trades_df['pnl']>0).mean() if len(trades_df)>0 else np.nan
    }
    return eq_series, trades_df, summary


# --- Streamlit UI ---
st.set_page_config(page_title="AI TraderX - Streamlit Demo", layout='wide')
st.title("AI TraderX — Streamlit Demo")
st.markdown("This is a single-file demo app for an AI-based trading project. It trains a simple XGBoost signal and runs a next-bar backtest (paper simulation). Use for college demo / learning only.")

# Sidebar: data source
st.sidebar.header("Data & Model Settings")
use_upload = st.sidebar.checkbox("Upload CSV instead of download", value=False)
if use_upload:
    uploaded = st.sidebar.file_uploader("Upload OHLC CSV (must have Date/Datetime, Open, High, Low, Close, Volume)", type=['csv'])
else:
    ticker = st.sidebar.text_input('Ticker (yfinance)', value='INFY.NS')
    period = st.sidebar.selectbox('Period', ['60d','180d','1y','2y','5y'], index=2)
    interval = st.sidebar.selectbox('Interval', ['1d','1h','30m','15m','5m'], index=0)

forward_bars = st.sidebar.number_input('Forward bars (target horizon)', min_value=1, max_value=60, value=1)
threshold = st.sidebar.number_input('Target return threshold (decimal)', min_value=0.0, max_value=0.1, value=0.004, step=0.001)
slippage = st.sidebar.number_input('Slippage pct (decimal)', min_value=0.0, max_value=0.01, value=0.0005, step=0.0001)
train_button = st.sidebar.button('Load & Compute / Train Model')

# Main layout
col1, col2 = st.columns([2,1])

with col1:
    st.subheader('Market Data')
    if use_upload:
        if uploaded is not None:
            try:
                df_raw = pd.read_csv(uploaded, parse_dates=True, infer_datetime_format=True)
                if 'Date' in df_raw.columns:
                    df_raw['datetime'] = pd.to_datetime(df_raw['Date'])
                    df_raw.set_index('datetime', inplace=True)
                elif 'Datetime' in df_raw.columns:
                    df_raw['datetime'] = pd.to_datetime(df_raw['Datetime'])
                    df_raw.set_index('datetime', inplace=True)
                else:
                    df_raw.index = pd.to_datetime(df_raw.index)
                df = df_raw[['Open','High','Low','Close','Volume']].copy()
                st.success('CSV loaded')
                st.dataframe(df.tail(10))
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")
                df = pd.DataFrame()
        else:
            st.info('Upload a CSV to begin')
            df = pd.DataFrame()
    else:
        if st.sidebar.button('Download Data'):
            df = download_data(ticker, period, interval)
            if df.empty:
                st.error('No data downloaded. Try a different ticker/interval')
            else:
                st.success(f'Downloaded {len(df)} rows for {ticker}')
                st.dataframe(df.tail(10))
        else:
            df = pd.DataFrame()

    if 'df' in locals() and not df.empty:
        st.subheader('Indicators & Features')
        df_features = compute_indicators(df)
        st.dataframe(df_features.tail(10))

        if train_button:
            df_target = make_target(df_features, forward_bars=forward_bars, threshold=threshold)
            features = ['SMA_10','SMA_50','EMA_21','RSI_14','ATR_14','return_1']
            st.info('Training XGBoost... this can take 30s-2m depending on data size')
            model, metrics = train_xgb(df_target, features)
            st.success('Training complete')
            st.write('Cross-validation metrics (lists by fold):')
            st.json(metrics)
            # save model
            os.makedirs('models', exist_ok=True)
            joblib.dump(model, 'models/xgb_signal.pkl')
            st.download_button('Download model (pkl)', data=joblib.dumps(model), file_name='xgb_signal.pkl')

            st.subheader('Backtest (paper)')
            eq_series, trades_df, summary = backtest(df_target, model, features, forward_bars=forward_bars, slippage_pct=slippage)
            st.metric('Starting Capital', f"{summary['starting_capital']:.2f}")
            st.metric('Ending Capital', f"{summary['ending_capital']:.2f}")
            st.metric('Net Profit', f"{summary['net_profit']:.2f}")
            st.metric('Total Trades', f"{summary['total_trades']}")
            st.metric('Win Rate', f"{summary['win_rate']:.2%}" if not np.isnan(summary['win_rate']) else 'N/A')

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq_series.index, y=eq_series.values, name='Equity'))
            fig.update_layout(title='Equity Curve (Paper Sim)', xaxis_title='Time', yaxis_title='Equity')
            st.plotly_chart(fig, use_container_width=True)

            if not trades_df.empty:
                st.subheader('Trades (sample)')
                trades_df_display = trades_df.copy()
                trades_df_display['entry_time'] = pd.to_datetime(trades_df_display['entry_time'])
                trades_df_display['exit_time'] = pd.to_datetime(trades_df_display['exit_time'])
                st.dataframe(trades_df_display.head(50))

with col2:
    st.subheader('Quick Strategy Builder')
    st.markdown('You can also run a simple rule-based strategy for comparison')
    rule = st.selectbox('Rule', ['SMA crossover','RSI oversold'])
    if st.button('Run Rule Backtest'):
        if 'df' in locals() and not df.empty:
            dff = compute_indicators(df)
            if rule == 'SMA crossover':
                dff['rule_signal'] = ((dff['SMA_10'] > dff['SMA_50']).astype(int))
                trades = []
                capital = 100000.0
                equity = capital
                for i in range(len(dff)-1):
                    if dff['rule_signal'].iloc[i] == 1 and dff['rule_signal'].iloc[i-1] == 0:
                        entry = dff['Open'].iloc[i+1]
                        exit = dff['Close'].iloc[i+1]
                        if pd.isna(entry) or pd.isna(exit):
                            continue
                        ret = (exit-entry)/entry
                        pnl = equity * 0.01 * ret
                        equity += pnl
                        trades.append(pnl)
                st.metric('Rule Ending Capital', f"{equity:.2f}")
            else:
                dff['rule_signal'] = (dff['RSI_14'] < 30).astype(int)
                # run similar small backtest
                st.write('RSI rule run (summary):')
                st.write(dff['rule_signal'].value_counts())
        else:
            st.error('Load data first')

    st.subheader('Model Explainability (optional)')
    if st.button('Show SHAP (if installed & model exists)'):
        try:
            import shap
            if os.path.exists('models/xgb_signal.pkl'):
                model = joblib.load('models/xgb_signal.pkl')
                if 'df_target' in locals():
                    Xsample = df_target[features].iloc[-200:]
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(Xsample)
                    st.header('SHAP summary (sample)')
                    st.pyplot(shap.summary_plot(shap_values, Xsample, show=False))
                else:
                    st.error('Train model first')
            else:
                st.error('No saved model found. Train a model first to view SHAP')
        except Exception as e:
            st.error(f'SHAP not available or failed: {e}')

st.markdown('---')
st.markdown('**Notes:** This demo is for educational and college project use. Do NOT use it for live trading without proper testing, compliance, and broker approval.')
