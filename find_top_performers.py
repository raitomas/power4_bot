
import sys
import os
import MetaTrader5 as mt5
import pandas as pd

sys.path.append(os.getcwd())

from backtesting.engine import BacktestEngine
from core.data_fetcher import descargar_ohlc

def main():
    if not mt5.initialize():
        print("Error MT5")
        return

    # Pool de 20 activos variados
    pool = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
        'SP500', 'NAS100', 'GER40', 'XAUUSD', 'XTIUSD',
        'BTCUSD', 'ETHUSD', 'AAPL', 'MSFT', 'TSLA', 
        'NVDA', 'AMZN', 'GOOGL', 'META', 'NFLX'
    ]
    
    # Intentar normalizar nombres según el broker
    all_syms = [s.name for s in mt5.symbols_get()]
    valid_pool = []
    for p in pool:
        for s in all_syms:
            if p in s:
                valid_pool.append(s)
                break
    
    valid_pool = list(set(valid_pool))[:20]
    print(f"Escaneando pool de {len(valid_pool)} activos...")

    engine = BacktestEngine(capital=100000, riesgo_pct=0.005)
    
    top_results = []
    for sym in valid_pool:
        df_d1 = descargar_ohlc(sym, "D1", n_barras=1600)
        df_w1 = descargar_ohlc(sym, "W1", n_barras=350)
        
        if df_d1 is not None and len(df_d1) > 400:
            print(f"Probando {sym}...")
            res = engine.ejecutar(sym, df_d1, df_w1)
            if res.total_trades > 0:
                top_results.append(res)
        
    # Ordenar por P&L
    top_results.sort(key=lambda x: x.pnl_total, reverse=True)

    print("\n" + "="*60)
    print(f"{'TOP PERFORMERS (MÉTODO POWER 4 - 5 AÑOS)':^60}")
    print("="*60)
    for res in top_results[:10]:
        print(f"{res.symbol:<10} | {res.total_trades:<3} trades | WR: {res.win_rate:>5.1%} | PF: {res.profit_factor:>4.2f} | P&L: ${res.pnl_total:>8,.0f}")
    print("="*60)

    mt5.shutdown()

if __name__ == "__main__":
    main()
