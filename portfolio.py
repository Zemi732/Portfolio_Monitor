import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from decimal import Decimal, ROUND_HALF_UP
import datetime
from datetime import timedelta

st.set_page_config(layout="wide", page_title="Portfolio Dashboard")

# ==========================================
# 0. CONFIGURATION: FEES (MER)
# ==========================================
MER_RATES = {
    'IVV': 0.04, 'IWDA': 0.20, 'BGBL': 0.08, 'VAS': 0.10, 'QSML': 0.35,
    'EMXC': 0.25, 'VGS': 0.18, 'VVLU': 0.28, 'ATOM': 0.69, 'SEMI': 0.45,
    'WIRE': 0.65, 'NDQ': 0.48, 'VUAA':0.07, 'XUSE': 0.15, 'EXCH':0.18, 
    'HACK': 0.67, 'PMGOLD': 0.00, 'WOW': 0.00, 'CSL': 0.00, 'BHP': 0.00,
    'CBA': 0.00, 'GYG': 0.00, 'QOR': 0.00
}

# ==========================================
# 1. SIDEBAR: FETCH & EDIT FX RATES
# ==========================================
with st.sidebar:
    st.header("Global Settings")

    @st.cache_data
    def get_exchange_rates():
        tickers = ['AUDUSD=X', 'AUDEUR=X', 'AUDPLN=X']
        try:
            data = yf.download(tickers, period="1d", progress=False)['Close'].iloc[-1]
            return pd.DataFrame({
                'Pair': ['AUD/USD', 'AUD/EUR', 'AUD/PLN'],
                'Rate': [data['AUDUSD=X'], data['AUDEUR=X'], data['AUDPLN=X']]
            })
        except:
            return pd.DataFrame({
                'Pair': ['AUD/USD', 'AUD/EUR', 'AUD/PLN'],
                'Rate': [0.65, 0.60, 2.65]
            })

    fx_data = get_exchange_rates()

    st.subheader("Exchange Rates (Live)")
    updated_fx = st.data_editor(
        fx_data,
        key="fx_sidebar",
        hide_index=True,
        column_config={"Rate": st.column_config.NumberColumn(format="%.4f")}
    )

    aud_usd = updated_fx.loc[0, "Rate"]
    
    # --- FX STATUS BAR ---
    if aud_usd > 0.70:
        usd_color = "#b71c1c" # Dark Red
        status_text = "STRONG BUY IVV/IWDA"
        action_note = "AUD is very strong. Excellent time to buy USD assets."
    elif aud_usd >= 0.68:
        usd_color = "#c62828" # Red
        status_text = "BUY IVV/IWDA"
        action_note = "AUD is strong. Good time to accumulate USD assets."
    elif aud_usd >= 0.65:
        usd_color = "#EF6C00" # Orange
        status_text = "NEUTRAL"
        action_note = "Market is in equilibrium."
    elif aud_usd >= 0.62:
        usd_color = "#2e7d32" # Green
        status_text = "BUY VAS / HEDGED"
        action_note = "AUD is weak. Prefer ASX or Hedged ETFs."
    else: 
        usd_color = "#1b5e20" # Dark Green
        status_text = "STRONG BUY VAS / HEDGED"
        action_note = "AUD is very weak. Avoid converting cash to USD."

    st.markdown(f"""
    <div style="background-color: {usd_color}; color: white; padding: 12px; border-radius: 8px; text-align: center; margin-top: 10px; border: 1px solid rgba(255,255,255,0.2);">
        <div style="font-size: 14px; opacity: 0.9;">Signal: <strong>{status_text}</strong></div>
        <div style="font-size: 24px; font-weight: bold; margin: 5px 0;">{aud_usd:.4f} USD</div>
        <div style="font-size: 13px; font-style: italic; border-top: 1px solid rgba(255,255,255,0.3); padding-top: 5px; margin-top: 5px;">
            {action_note}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Refresh Prices"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")

# ==========================================
# 2. MAIN PAGE: DATA LOADING & LOGIC
# ==========================================
st.title("🇦🇺 ASX & 🇺🇸 US Portfolio Monitor")

CORE_ORDER = ['VUAA', 'XUSE', 'EXCH', 'BGBL', 'VAS', 'EMXC', 'QSML']
CORE_TICKERS = ['VUAA', 'XUSE', 'EXCH', 'BGBL', 'VAS', 'EMXC', 'QSML', 'IWDA']
US_TICKERS = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'TSLA', 'PLTR'] 

# --- DEEP GEOGRAPHIC MAPPING (X-RAY) ---
GEO_MAP = {
    'VAS': {'Australia': 1.0}, 'BHP': {'Australia': 1.0}, 'CBA': {'Australia': 1.0}, 
    'CSL': {'Australia': 1.0}, 'WOW': {'Australia': 1.0}, 'GYG': {'Australia': 1.0}, 
    'QOR': {'Australia': 1.0}, 'XRO': {'Australia': 1.0}, 'PMGOLD': {'Australia': 1.0},
    
    'NVDA': {'United States': 1.0}, 'MSFT': {'United States': 1.0}, 'AAPL': {'United States': 1.0}, 
    'AMZN': {'United States': 1.0}, 'TSLA': {'United States': 1.0}, 'PLTR': {'United States': 1.0}, 
    'IVV': {'United States': 1.0}, 'VUAA': {'United States': 1.0},
    
    'IWDA': {'United States': 0.72, 'Japan': 0.06, 'United Kingdom': 0.04, 'France': 0.03, 'Canada': 0.03, 'Rest of World': 0.12},
    'XUSE': {'Japan': 0.19, 'United Kingdom': 0.13, 'Canada': 0.12, 'France': 0.09, 'Germany': 0.08, 'Australia': 0.06, 'Rest of World': 0.33},
    'BGBL': {'United States': 0.72, 'Japan': 0.06, 'United Kingdom': 0.04, 'France': 0.03, 'Canada': 0.03, 'Rest of World': 0.12},
    'VGS': {'United States': 0.72, 'Japan': 0.06, 'United Kingdom': 0.04, 'France': 0.03, 'Canada': 0.03, 'Rest of World': 0.12},
    'VVLU': {'United States': 0.65, 'Japan': 0.10, 'United Kingdom': 0.05, 'Rest of World': 0.20},
    'QSML': {'United States': 0.60, 'Japan': 0.10, 'United Kingdom': 0.05, 'Rest of World': 0.25},
    
    'EMXC': {'India': 0.25, 'Taiwan': 0.25, 'South Korea': 0.15, 'Brazil': 0.05, 'Rest of World': 0.30},
    'EXCH': {'Taiwan': 0.29, 'South Korea': 0.21, 'India': 0.18, 'Brazil': 0.06, 'South Africa': 0.05, 'Rest of World': 0.21},
    
    'ATOM': {'United States': 0.70, 'Rest of World': 0.30}, 
    'SEMI': {'United States': 0.70, 'Taiwan': 0.15, 'Rest of World': 0.15}, 
    'WIRE': {'United States': 0.50, 'Europe': 0.30, 'Rest of World': 0.20}
}
# ---------------------------------------

FX_SENSITIVITY = {
    'IWDA': 'Very High', 'XUSE': 'Very High', 'EXCH': 'Very High', 'VUAA': 'Very High', 'IVV': 'Very High', 'SEMI': 'Very High', 
    'ATOM': 'Very High', 'WIRE': 'Very High', 'BHP': 'Very High', 'CSL': 'Very High', 'NDQ': 'Very High', 'HACK': 'Very High',
    'VGS': 'High', 'BGBL': 'High', 'QSML': 'High', 'EMXC': 'High', 
    'QOR': 'High', 'PMGOLD': 'High', 'XRO': 'High',
    'VAS': 'Medium', 'VVLU': 'Medium',
    'WOW': 'Low', 'GYG': 'Low', 'CBA': 'Very Low'
}

def get_fx_tilt(ticker, aud_usd):
    sens = FX_SENSITIVITY.get(ticker, 'Medium')
    if aud_usd > 0.70:
        return "Strong Buy" if sens in ['Very High', 'High'] else "Buy" if sens == 'Medium' else "Hold"
    elif aud_usd >= 0.68:
        return "Buy" if sens in ['Very High', 'High'] else "Hold" if sens == 'Medium' else "Sell"
    elif aud_usd >= 0.65:
        return "Neutral"
    elif aud_usd >= 0.62:
        return "Buy" if sens in ['Low', 'Very Low'] else "Hold" if sens == 'Medium' else "Sell" 
    else:
        return "Strong Buy" if sens in ['Low', 'Very Low'] else "Buy" if sens == 'Medium' else "Strong Sell"

# --- A. LOAD HOLDINGS & CALCULATE METRICS ---
def load_data():
    total_brokerage_paid = Decimal('0')
    total_realized_pl_lifetime = Decimal('0')
    
    try:
        # ---> LIVE HOLDINGS CONNECTION <---
        sheet_url = "https://docs.google.com/spreadsheets/d/1yzFLgUMXo0iutBoEEEEstJl5EcXHHpKu6EG082fEWGI/export?format=csv"
        df_trades = pd.read_csv(sheet_url)
        # ----------------------------------
        df_trades.columns = df_trades.columns.str.strip()
        df_trades['Trade Date'] = pd.to_datetime(df_trades['Trade Date'], dayfirst=True, format='mixed')
        df_trades = df_trades.sort_values('Trade Date', ascending=True)
        holdings_dict = {}
        
        for ticker, group in df_trades.groupby('Instrument Code'):
            total_shares_calc = Decimal('0')
            total_cost = Decimal('0')
            ticker_realized_pl = Decimal('0')
            
            for index, row in group.iterrows():
                trans_type = str(row['Transaction Type']).lower()
                qty = Decimal(str(row['Quantity']))
                price = Decimal(str(row['Price']))
                
                trade_brokerage = Decimal('0')
                if 'Brokerage' in row and pd.notnull(row['Brokerage']): raw_b = str(row['Brokerage'])
                elif 'Commission' in row and pd.notnull(row['Commission']): raw_b = str(row['Commission'])
                elif 'Fee' in row and pd.notnull(row['Fee']): raw_b = str(row['Fee'])
                else: raw_b = ''
                
                try: 
                    clean_b = raw_b.replace('$','').replace(',','').replace('-','')
                    if clean_b.strip(): trade_brokerage = Decimal(clean_b)
                except: trade_brokerage = Decimal('0')
                
                total_brokerage_paid += trade_brokerage

                if 'buy' in trans_type:
                    total_shares_calc += qty
                    total_cost += (qty * price) + trade_brokerage
                
                elif 'sell' in trans_type:
                    if total_shares_calc > 0:
                        avg_c = total_cost / total_shares_calc
                        cost_basis_sold = qty * avg_c
                        proceeds = (qty * price) - trade_brokerage
                        trade_pl = proceeds - cost_basis_sold
                        ticker_realized_pl += trade_pl
                        total_realized_pl_lifetime += trade_pl
                        total_cost -= cost_basis_sold
                        total_shares_calc -= qty

            if total_shares_calc > 0:
                avg_price_dec = total_cost / total_shares_calc
                avg_price = float(avg_price_dec)
                total_shares = float(total_shares_calc)
                
                if ticker in CORE_TICKERS: cat = "Core"
                elif ticker in US_TICKERS: cat = "US Market"
                else: cat = "Satellite"
                
                holdings_dict[ticker] = {
                    'Ticker': ticker, 'Category': cat, 'Shares': total_shares, 
                    'Avg_Price': avg_price, 'Realized_PL_Active': float(ticker_realized_pl)
                }

        if not holdings_dict: return pd.DataFrame(), 0.0, 0.0
        return pd.DataFrame.from_dict(holdings_dict, orient='index'), float(total_brokerage_paid), float(total_realized_pl_lifetime)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), 0.0, 0.0

# --- B. FETCH PRICES ---
TICKER_MAP = {
    "MSFT": "MSFT",
    "EXCH": "EXCH.AS",
    "VUAA": "VUAA.L",
    "XUSE": "XUSE.SW",
    "PMGOLD": "PMGOLD.AX",
    "IWDA": "IWDA.L" 
}

@st.cache_data(ttl=300)
def fetch_market_data(ticker_list):
    prices = {}
    fx_multipliers = {}
    
    try:
        usd_aud_rate = yf.Ticker("USDAUD=X").fast_info['last_price']
    except:
        usd_aud_rate = 1.0 
        
    for t in ticker_list:
        if t in TICKER_MAP:
            yf_ticker = TICKER_MAP[t]
        elif 'US_TICKERS' in globals() and t in US_TICKERS:
            yf_ticker = t
        else:
            yf_ticker = f"{t}.AX"
            
        try:
            ticker_obj = yf.Ticker(yf_ticker)
            prices[t] = float(ticker_obj.fast_info['last_price'])
            
            if yf_ticker.endswith('.AX'):
                fx_multipliers[t] = 1.0
            else:
                fx_multipliers[t] = usd_aud_rate
                
        except Exception as e:
            prices[t] = 0.0
            fx_multipliers[t] = 1.0
            
    return prices, fx_multipliers
    
# --- FETCH EARNINGS DATES ---
@st.cache_data(ttl=3600*12) 
def fetch_earnings_dates(ticker_list):
    earnings_map = {}
    for t in ticker_list:
        if t in US_TICKERS: search_t = t
        else: search_t = f"{t}.AX"
        
        try:
            stock = yf.Ticker(search_t)
            cal = stock.calendar
            if cal and 'Earnings Date' in cal:
                dates = cal['Earnings Date']
                if dates:
                    next_date = dates[0]
                    earnings_map[t] = next_date.date() 
                else:
                    earnings_map[t] = None
            else:
                earnings_map[t] = None
        except:
            earnings_map[t] = None
            
    return earnings_map

# --- C. STYLING ---
def apply_portfolio_styling(dataframe, price_col_name, avg_col_name='Avg_Price'):
    format_dict = {
        avg_col_name: '{:.2f}', price_col_name: '{:.2f}', 
        'P/L %': '{:+.1f}%', 'P/L (AUD)': '{:+.2f}', 'Realized P/L': '{:+.2f}',
        'Market_Value_AUD': '${:,.0f}', 'Distribution': '{:.1f}%', 
        'Target_%': '{:.0f}%', 'Shares': '{:.0f}', 'Deficit': '${:,.0f}'
    }
    
    def custom_styler(row):
        styles = ['' for _ in row.index]
        if row['Ticker'] == 'Satellites': styles = ['font-weight: bold' for _ in row.index]
        
        # P/L %
        try:
            if 'P/L %' in row and pd.notnull(row['P/L %']):
                rise_val = row['P/L %']
                if rise_val >= 0: rise_bg = 'background-color: #2e7d32; color: white; font-weight: bold'
                elif rise_val > -5.0: rise_bg = 'background-color: #EF6C00; color: white; font-weight: bold'
                else: rise_bg = 'background-color: #c62828; color: white; font-weight: bold'
                if 'P/L %' in row.index: styles[list(row.index).index('P/L %')] = rise_bg
        except: pass

        # P/L (AUD)
        try:
            if 'P/L (AUD)' in row and pd.notnull(row['P/L (AUD)']):
                prof_val = row['P/L (AUD)']
                if prof_val >= 0: prof_style = 'color: #2e7d32; font-weight: bold' 
                else: prof_style = 'color: #c62828; font-weight: bold' 
                if 'P/L (AUD)' in row.index: styles[list(row.index).index('P/L (AUD)')] = prof_style
        except: pass

        # Realized P/L
        try:
            if 'Realized P/L' in row and pd.notnull(row['Realized P/L']):
                real_val = row['Realized P/L']
                if real_val > 0: real_style = 'color: #2e7d32; font-weight: bold' 
                elif real_val < 0: real_style = 'color: #c62828; font-weight: bold' 
                else: real_style = 'color: white'
                if 'Realized P/L' in row.index: styles[list(row.index).index('Realized P/L')] = real_style
        except: pass

        # FX Tilt
        try:
            if 'FX Tilt' in row:
                tilt = row['FX Tilt']
                tilt_bg = ''
                if tilt == 'Strong Buy': tilt_bg = 'background-color: #b71c1c; color: white; font-weight: bold'
                elif tilt == 'Buy': tilt_bg = 'background-color: #c62828; color: white; font-weight: bold'
                elif tilt == 'Neutral': tilt_bg = 'background-color: #EF6C00; color: white; font-weight: bold'
                elif tilt == 'Sell': tilt_bg = 'background-color: #2e7d32; color: white; font-weight: bold'
                elif tilt == 'Strong Sell': tilt_bg = 'background-color: #1b5e20; color: white; font-weight: bold'
                if 'FX Tilt' in row.index: styles[list(row.index).index('FX Tilt')] = tilt_bg
        except: pass

        # Deficit
        try:
            if 'Deficit' in row and pd.notnull(row['Deficit']):
                def_val = row['Deficit']
                if def_val > 50: def_style = 'color: #2e7d32; font-weight: bold' 
                elif def_val < -50: def_style = 'color: #c62828; font-weight: bold' 
                else: def_style = 'color: white'
                if 'Deficit' in row.index: styles[list(row.index).index('Deficit')] = def_style
        except: pass
        
        # Next Earnings (RED if < 7 days)
        try:
            if 'Next Earnings' in row:
                earnings_date = row['Next Earnings']
                if isinstance(earnings_date, (datetime.date, datetime.datetime)):
                    today = datetime.date.today()
                    if isinstance(earnings_date, datetime.datetime): earnings_date = earnings_date.date()
                    delta = (earnings_date - today).days
                    if 0 <= delta < 7:
                        earn_style = 'background-color: #c62828; color: white; font-weight: bold'
                        if 'Next Earnings' in row.index: styles[list(row.index).index('Next Earnings')] = earn_style
        except: pass

        # Distribution Logic
        try:
            if 'Distribution' in row and pd.notnull(row['Distribution']):
                dist_val = row['Distribution']
                if row['Ticker'] == 'Satellites':
                    if dist_val > 10.0:
                        dist_style = 'color: #c62828; font-weight: bold' 
                        if 'Distribution' in row.index: styles[list(row.index).index('Distribution')] = dist_style
                elif row['Ticker'] not in CORE_ORDER: 
                     if dist_val > 5.0:
                        dist_style = 'color: #c62828; font-weight: bold' 
                        if 'Distribution' in row.index: styles[list(row.index).index('Distribution')] = dist_style
        except: pass
        
        return styles
    return dataframe.style.format(format_dict).apply(custom_styler, axis=1)

# --- D. MAIN EXECUTION ---
df, total_brokerage_val, total_lifetime_realized = load_data()

if not df.empty:
    # 1. Fetch live prices & FX rates
    ticker_list = df['Ticker'].tolist()
    with st.spinner('Fetching market prices...'):
        current_prices, fx_multipliers = fetch_market_data(ticker_list)
    
    with st.spinner('Checking earnings calendars...'):
        earnings_dates = fetch_earnings_dates(ticker_list)

    api_missing_tickers = [t for t, p in current_prices.items() if p <= 0]
    if "manual_prices_storage" not in st.session_state: st.session_state["manual_prices_storage"] = {}
    for t in api_missing_tickers:
        if t in st.session_state["manual_prices_storage"]: current_prices[t] = st.session_state["manual_prices_storage"][t]

    # 2. Map standard data
    df['Current_Price'] = df['Ticker'].map(current_prices)
    df['FX Rate'] = df['Ticker'].map(fx_multipliers)
    df['Next Earnings'] = df['Ticker'].map(earnings_dates) 
    
    # 3. Calculate Wealth Metrics (Normalized to AUD)
    df['Rise'] = ((df['Current_Price'] - df['Avg_Price']) / df['Avg_Price']) * 100
    df['FX Tilt'] = df['Ticker'].apply(lambda t: get_fx_tilt(t, aud_usd))
    
    df['Market_Value_AUD'] = df['Shares'] * df['Current_Price'] * df['FX Rate']
    df['Gain_Loss_Native'] = (df['Current_Price'] - df['Avg_Price']) * df['Shares'] * df['FX Rate']
    df['Realized_PL_Active'] = df['Realized_PL_Active'] * df['FX Rate']
    
    total_value_aud = df['Market_Value_AUD'].sum() 
    
    # --- TARGETS SIDEBAR ---
    st.sidebar.title("🎯 Targets")
    targets = {}
    PRESETS = {'VUAA': 50, 'XUSE': 30, 'VAS': 10, 'EXCH': 10, 'EMXC': 0, 'QSML': 0}
    st.sidebar.markdown("### 🟢 Core")
    for ticker in CORE_ORDER:
        if ticker in df['Ticker'].values:
            targets[ticker] = st.sidebar.slider(f"{ticker}", 0, 100, PRESETS.get(ticker, 0))
    st.sidebar.markdown("### 📦 Satellites")
    sat_target_total = st.sidebar.slider("Satellite Fund %", 0, 100, 10)
    df['Target_%'] = df['Ticker'].map(targets).fillna(0)
    df['Target_Value'] = total_value_aud * (df['Target_%'] / 100)
    df['Deficit'] = df['Target_Value'] - df['Market_Value_AUD']

    # --- AGGREGATE SATELLITES ROW ---
    df_sat = df[df['Category'] == 'Satellite'].copy()
    agg_row = None
    if not df_sat.empty:
        sat_val = df_sat['Market_Value_AUD'].sum()
        if sat_val > 0:
            agg_row = {
                'Ticker': 'Satellites', 'Shares': df_sat['Shares'].mean(),
                'Avg_Price': (df_sat['Market_Value_AUD']*df_sat['Avg_Price']).sum()/sat_val,
                'Current_Price': (df_sat['Market_Value_AUD']*df_sat['Current_Price']).sum()/sat_val,
                'Rise': (df_sat['Market_Value_AUD']*df_sat['Rise']).sum()/sat_val,
                'Market_Value_AUD': sat_val, 'Target_%': sat_target_total, 
                'Deficit': (total_value_aud*sat_target_total/100) - sat_val,
                'Profit': df_sat['Gain_Loss_Native'].sum(), 
                'Realized_PL_Active': df_sat['Realized_PL_Active'].sum(),
                'FX Tilt': 'Neutral',
                'Next Earnings': None 
            }

    # --- PORTFOLIO HEADER METRICS ---
    total_profit_aud = df['Gain_Loss_Native'].sum()
    total_cost_aud = total_value_aud - total_profit_aud
    total_return_pct = (total_profit_aud / total_cost_aud) * 100 if total_cost_aud > 0 else 0.0
    
    profit_color = "#2e7d32" if total_profit_aud >= 0 else "#c62828"
    profit_sign = "+" if total_profit_aud >= 0 else "-"
    pct_color = "#2e7d32" if total_return_pct >= 0 else "#c62828"
    pct_sign = "+" if total_return_pct >= 0 else ""

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Value (AUD)", f"${total_value_aud:,.2f}")
    c2.markdown(f"""<div style="margin-top: 5px;"><p style="font-size: 14px; margin-bottom: 0px; opacity: 0.8;">Est. Total Profit (AUD)</p><p style="font-size: 30px; font-weight: bold; color: {profit_color}; margin: 0px;">{profit_sign}${abs(total_profit_aud):,.2f}</p></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div style="margin-top: 5px;"><p style="font-size: 14px; margin-bottom: 0px; opacity: 0.8;">Total Profit %</p><p style="font-size: 30px; font-weight: bold; color: {pct_color}; margin: 0px;">{pct_sign}{total_return_pct:.2f}%</p></div>""", unsafe_allow_html=True)
    
    # --- TABLE SPLITTING & RENAMING ---
    df_core = df[df['Category'] == 'Core'].copy()
    if not df_core.empty:
        if 'CORE_ORDER' in globals():
            df_core['Ticker'] = pd.Categorical(df_core['Ticker'], categories=CORE_ORDER, ordered=True)
        df_core = df_core.sort_values('Ticker')
        if 'agg_row' in globals() and agg_row: 
            df_core = pd.concat([df_core, pd.DataFrame([agg_row])], ignore_index=True)
            
        df_core['Distribution'] = (df_core['Market_Value_AUD'] / total_value_aud) * 100
        df_core['Profit'] = df_core['Gain_Loss_Native']
        df_core['Realized P/L'] = df_core['Realized_PL_Active']
        df_core = df_core.rename(columns={
            'Rise': 'P/L %', 
            'Profit': 'P/L (AUD)', 
            'Current_Price': 'ASX Price'
        })

    df_us = df[df['Category'] == 'US Market'].copy()
    if not df_us.empty: 
        df_us['Distribution'] = (df_us['Market_Value_AUD'] / total_value_aud) * 100
        df_us['Profit'] = df_us['Gain_Loss_Native']
        df_us['Realized P/L'] = df_us['Realized_PL_Active']
        df_us = df_us.rename(columns={
            'Rise': 'P/L %', 
            'Profit': 'P/L (AUD)', 
            'Current_Price': 'Price (USD)', 
            'Avg_Price': 'Avg Price (USD)'
        })

    df_sat = df[df['Category'] == 'Satellite'].copy()
    if not df_sat.empty: 
        df_sat = df_sat.sort_values('Rise', ascending=False)
        df_sat['Distribution'] = (df_sat['Market_Value_AUD'] / total_value_aud) * 100
        df_sat['Profit'] = df_sat['Gain_Loss_Native']
        df_sat['Realized P/L'] = df_sat['Realized_PL_Active']
        df_sat = df_sat.rename(columns={
            'Rise': 'P/L %', 
            'Profit': 'P/L (AUD)', 
            'Current_Price': 'Price (USD)',
            'Avg_Price': 'Avg Price (USD)'
        })

    # --- TABLE DISPLAY ---
    st.subheader("🪐 Core Portfolio (With Fund)")
    if not df_core.empty:
        cols_core = ['Ticker', 'Shares', 'Avg_Price', 'ASX Price', 'P/L %', 'P/L (AUD)', 'Realized P/L', 'FX Tilt', 'Market_Value_AUD', 'Target_%', 'Deficit', 'Distribution']
        st.dataframe(apply_portfolio_styling(df_core[cols_core], 'ASX Price'))

    st.subheader("🛰️ Satellite Fund Composition")
    if not df_sat.empty:
        cols_sat = ['Ticker', 'Shares', 'Avg Price (USD)', 'Price (USD)', 'P/L %', 'P/L (AUD)', 'Realized P/L', 'FX Tilt', 'Next Earnings', 'Market_Value_AUD', 'Distribution']
        st.dataframe(apply_portfolio_styling(df_sat[cols_sat], 'Price (USD)'))

    st.subheader("🦅 US Market")
    if not df_us.empty:
        cols_us = ['Ticker', 'Shares', 'Avg Price (USD)', 'Price (USD)', 'P/L %', 'P/L (AUD)', 'Realized P/L', 'FX Tilt', 'Next Earnings', 'Market_Value_AUD', 'Distribution']
        st.dataframe(apply_portfolio_styling(df_us[cols_us], 'Price (USD)', 'Avg Price (USD)'))
    else: 
        st.info("No US holdings.")
        
    if api_missing_tickers:
        st.divider()
        st.warning(f"⚠️ Manual Input Required: {len(api_missing_tickers)} prices could not be fetched.")
        init_data = [{'Ticker': t, 'Manual_Price': st.session_state["manual_prices_storage"].get(t, 0.0)} for t in api_missing_tickers]
        df_missing_input = pd.DataFrame(init_data)
        edited_df = st.data_editor(df_missing_input, key="manual_editor", hide_index=True, num_rows="fixed")
        if not edited_df.equals(df_missing_input):
            new_map = dict(zip(edited_df['Ticker'], edited_df['Manual_Price']))
            st.session_state["manual_prices_storage"].update(new_map)
            st.rerun()

    st.divider()
    c_pie1, c_pie2 = st.columns(2)
    
    with c_pie1:
        st.write("### Core Distribution")
        if not df_core.empty:
            fig = px.pie(df_core, values='Market_Value_AUD', names='Ticker', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0)); fig.update_traces(textinfo='label+percent')
            st.plotly_chart(fig, use_container_width=True)
            
    with c_pie2:
        st.write("### True Geographic Exposure")
        geo_breakdown = []
        for index, row in df.iterrows():
            ticker = row['Ticker']
            total_value = row['Market_Value_AUD']
            weights = GEO_MAP.get(ticker, {'Other': 1.0})
            for country, percentage in weights.items():
                geo_breakdown.append({'Country': country, 'Value': total_value * percentage})
                
        geo_df = pd.DataFrame(geo_breakdown)
        geo_summary = geo_df.groupby('Country', as_index=False)['Value'].sum()
        
        if not geo_summary.empty:
            geo_summary = geo_summary.sort_values(by='Value', ascending=False)
            fig_geo = px.pie(geo_summary, values='Value', names='Country', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
            fig_geo.update_layout(margin=dict(t=0, b=0, l=0, r=0)); fig_geo.update_traces(textinfo='label+percent')
            st.plotly_chart(fig_geo, use_container_width=True)

    with st.sidebar:
        st.header("💰 Net Worth")
        total_us_aud = df[df['Category']=='US Market']['Market_Value_AUD'].sum()
        total_sat_aud = df[df['Category']=='Satellite']['Market_Value_AUD'].sum()
        total_core_aud = df[df['Category']=='Core']['Market_Value_AUD'].sum()
        grand_total = total_us_aud + total_sat_aud + total_core_aud

        st.metric("Total Net Worth (AUD)", f"${grand_total:,.2f}")
        st.write("Breakdown (AUD):")
        st.caption(f"🟢 Core: ${total_core_aud:,.0f}")
        st.caption(f"🦅 US Stocks: ${total_us_aud:,.0f}")
        st.caption(f"🛰️ Satellite: ${total_sat_aud:,.0f}")

        # --- FEES SECTION ---
        st.divider()
        st.header("📉 Fees")
        
        df['MER'] = df['Ticker'].map(MER_RATES).fillna(0.00)
        df['Annual_Fee_AUD'] = df['Market_Value_AUD'] * (df['MER'] / 100)
        total_annual_fee = df['Annual_Fee_AUD'].sum()
        weighted_avg_mer = (total_annual_fee / grand_total) * 100 if grand_total > 0 else 0.00

        c_fee1, c_fee2 = st.columns(2)
        c_fee1.metric("Avg MER", f"{weighted_avg_mer:.2f}%")
        c_fee2.metric("Annual MER", f"${total_annual_fee:,.0f}")
        st.caption("Based on Market Value & MER estimates.")

        # --- BROKERAGE & REALIZED PL ---
        st.divider()
        st.metric("💸 Total Brokerage Paid", f"${total_brokerage_val:,.2f}")
        
        st.metric("💰 Total Realized P/L", f"${total_lifetime_realized:,.2f}", help="Lifetime realized profit/loss from all sold positions.")
        st.caption("Includes fully sold positions.")

# ---> SHOPPING LIST <---
st.subheader("🛒 Shopping List")
try:
    sheet_url = "https://docs.google.com/spreadsheets/d/1dBmx0FsTUKh0tOOFfLYnhq5iwlZYyZIJYZaxZj62orQ/export?format=csv"
    wish_list = pd.read_csv(sheet_url)

    wish_list = wish_list.dropna(subset=['Ticker'])
    wish_list['Ticker'] = wish_list['Ticker'].astype(str).str.strip()
    wish_list = wish_list[wish_list['Ticker'] != '']
    wish_list = wish_list[wish_list['Ticker'] != 'nan']

    wish_list.columns = wish_list.columns.str.strip()
    wish_list['Desired Price'] = wish_list['Desired Price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
    wish_list['Desired Price'] = pd.to_numeric(wish_list['Desired Price'], errors='coerce')
        
    actual_prices = []
    ws_targets = []
    year_lows = []
    year_highs = []
    pe_trailing = []
    pe_forward = []
    div_yields = []
       
    for ticker in wish_list['Ticker']:
        try:
            stock = yf.Ticker(ticker)
            
            try: actual_prices.append(stock.fast_info['last_price'])
            except: actual_prices.append(None)
            
            try: year_lows.append(stock.fast_info['year_low'])
            except: year_lows.append(None)
            
            try: year_highs.append(stock.fast_info['year_high'])
            except: year_highs.append(None)
            
            info = stock.info
            ws_targets.append(info.get('targetMeanPrice', None))
            pe_trailing.append(info.get('trailingPE', None))
            pe_forward.append(info.get('forwardPE', None))
            
            raw_yield = info.get('dividendYield')
            div_yields.append(raw_yield * 100 if raw_yield else None)

        except Exception as e:
            actual_prices.append(None)
            ws_targets.append(None)
            year_lows.append(None)
            year_highs.append(None)
            pe_trailing.append(None)
            pe_forward.append(None)
            div_yields.append(None)
            
    wish_list['Actual Price'] = actual_prices
    wish_list['% Diff'] = (wish_list['Actual Price'] - wish_list['Desired Price']) / wish_list['Desired Price']
    wish_list['WS Target'] = ws_targets
    wish_list['52W Low'] = year_lows
    wish_list['52W High'] = year_highs
    wish_list['Trailing P/E'] = pe_trailing
    wish_list['Forward P/E'] = pe_forward
    wish_list['Yield'] = div_yields

    wish_list = wish_list.sort_values(by='% Diff', ascending=True)
    
    def style_shopping_list(row):
        styles = [''] * len(row)
        actual = row['Actual Price']
        desired = row['Desired Price']
        
        if pd.isna(actual) or pd.isna(desired):
            return styles
            
        price_col_idx = row.index.get_loc('Actual Price')
        diff_col_idx = row.index.get_loc('% Diff')
        
        if actual <= desired: color = '#00FF00'
        elif actual <= (desired * 1.02): color = '#FFA500'
        else: color = '#FF0000'
            
        styles[price_col_idx] = f'color: {color}; font-weight: bold;'
        styles[diff_col_idx] = f'color: {color}; font-weight: bold;'
            
        return styles

    styled_wish_list = (
        wish_list.style
        .apply(style_shopping_list, axis=1)
        .format({
            '% Diff': '{:+.2%}', 
            'Actual Price': '${:.2f}', 
            'Desired Price': '${:.2f}',
            'WS Target': '${:.2f}',
            '52W Low': '${:.2f}',
            '52W High': '${:.2f}',
            'Trailing P/E': '{:.1f}',
            'Forward P/E': '{:.1f}',
            'Yield': '{:.2f}%'
        }, na_rep="-")
    )
    
    st.dataframe(styled_wish_list, use_container_width=True)

except FileNotFoundError:
    st.warning("Could not find 'Wish list.csv'. Please make sure it is saved in the exact same folder as your script.")

# ---> NEW SECTION: ASX 200 BARGAIN SCANNER <---
st.subheader("📉 ASX 200 Daily Losers (Bargain Scanner)")

@st.cache_data(ttl=3600)
def get_asx_losers():
    try:
        import requests
        
        # 1. Disguise the script as a normal web browser
        url = 'https://en.wikipedia.org/wiki/S%26P/ASX_200'
        header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        req = requests.get(url, headers=header)
        tables = pd.read_html(req.text)
        
        # 2. Dynamically hunt for the correct table (no matter where it moves)
        df_asx200 = None
        for table in tables:
            if 'Code' in table.columns and 'Company' in table.columns:
                df_asx200 = table
                break
                
        if df_asx200 is None:
            st.error("Error: Could not find the ASX 200 constituents table on Wikipedia.")
            return pd.DataFrame()
        
        tickers = df_asx200['Code'].astype(str) + '.AX'
        ticker_list = tickers.tolist()

        # 3. Fetch the data from Yahoo Finance
        data = yf.download(ticker_list, period="5d", progress=False)
        
        # Handle different versions of yfinance returning multi-level columns
        if 'Close' in data.columns:
            data = data['Close']
            
        if data.empty or len(data) < 2:
            st.error("Error: Not enough recent price data available from Yahoo Finance.")
            return pd.DataFrame()
        
        # 4. Calculate the drop
        recent_data = data.iloc[-2:] 
        pct_change = ((recent_data.iloc[1] - recent_data.iloc[0]) / recent_data.iloc[0]) * 100
        
        # 5. Grab the top 5 biggest drops
        losers = pct_change.sort_values().head(5)
        
        losers_df = pd.DataFrame({
            'Ticker': losers.index.str.replace('.AX', '', regex=False),
            'Company': df_asx200.set_index('Code').loc[losers.index.str.replace('.AX', '', regex=False)]['Company'].values,
            'Drop %': losers.values,
            'Last Price': recent_data.iloc[1][losers.index].values
        })
        return losers_df
        
    except Exception as e:
        # If it fails again, it will print the EXACT error message to your dashboard
        st.error(f"Bargain Scanner Error: {e}")
        return pd.DataFrame()

with st.spinner('Scanning ASX 200 for bargains...'):
    losers_df = get_asx_losers()

if not losers_df.empty:
    styled_losers = (
        losers_df.style
        .map(lambda x: 'color: #c62828; font-weight: bold;', subset=['Drop %'])
        .format({
            'Drop %': '{:.2f}%',
            'Last Price': '${:.2f}'
        })
    )
    st.dataframe(styled_losers, hide_index=True, use_container_width=True)
else:
    st.info("Market scanner currently unavailable. Check your connection.")
# ---> END NEW SECTION <---

# --- FINAL CATCH-ALL FOR EMPTY PORTFOLIO DATA ---
if df.empty:
    st.info("Waiting for data...")



