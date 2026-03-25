
import sys
import os
import yaml
import MetaTrader5 as mt5
import pandas as pd

# Añadir el path del proyecto para importar los módulos
sys.path.append(os.getcwd())

from backtesting.engine import BacktestEngine
from core.data_fetcher import descargar_ohlc

def main():
    if not mt5.initialize():
        print("Error al inicializar MetaTrader 5")
        return

    # Definir activos relevantes de diferentes categorías
    symbols = ['EURUSD', 'AAPL', 'BTC', 'SP500', 'XAUUSD']
    
    # Configurar motor de backtest: 100k capital, 0.5% riesgo por op
    engine = BacktestEngine(
        capital=100000.0, 
        riesgo_pct=0.005,
        min_barras=200
    )
    
    resultados = {}
    for sym in symbols:
        # Intentar descargar 1500 barras (aprox 6 años para tener margen de indicadores)
        df_d1 = descargar_ohlc(sym, "D1", n_barras=1600)
        df_w1 = descargar_ohlc(sym, "W1", n_barras=350)
        
        if df_d1 is not None and len(df_d1) > 400:
            print(f"Ejecutando backtest para {sym} ({len(df_d1)} velas)...")
            res = engine.ejecutar(sym, df_d1, df_w1)
            resultados[sym] = res
        else:
            print(f"Error o datos insuficientes para {sym}")

    # Imprimir resumen de alta fidelidad
    print("\n" + "="*60)
    print(f"{'RESUMEN EJECUTIVO DE BACKTESTING':^60}")
    print("="*60)
    print(f"{'Activo':<10} | {'Trades':<6} | {'WR':<6} | {'PF':<6} | {'MaxDD':<7} | {'P&L ($)':<10}")
    print("-" * 60)
    
    for sym, res in resultados.items():
        if res.total_trades > 0:
            print(f"{sym:<10} | {res.total_trades:<6} | {res.win_rate:>5.1%} | {res.profit_factor:>6.2f} | {res.max_drawdown_pct:>7.1%} | ${res.pnl_total:>10,.0f}")
        else:
            print(f"{sym:<10} | No se detectaron señales en el período.")

    print("="*60)
    mt5.shutdown()

if __name__ == "__main__":
    main()
