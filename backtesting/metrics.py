"""
power4_bot/backtesting/metrics.py
================================================
Generador de reportes y métricas del backtesting.
Produce tablas de resumen y datos para el dashboard.
================================================
"""

import logging
from typing import Dict, List
import pandas as pd
import numpy as np

from backtesting.engine import ResultadoBacktest, Trade

logger = logging.getLogger(__name__)


def resumen_global(resultados: Dict[str, ResultadoBacktest]) -> dict:
    """
    Agrega los resultados de todos los activos en un
    resumen global de rendimiento.
    """
    todos_trades: List[Trade] = []
    for r in resultados.values():
        todos_trades.extend(r.trades)

    if not todos_trades:
        return {}

    total    = len(todos_trades)
    ganadoras = sum(1 for t in todos_trades if t.ganadora)
    pnl_total = sum(t.pnl_dolares for t in todos_trades)
    ganancias = sum(t.pnl_dolares for t in todos_trades if t.ganadora)
    perdidas  = abs(sum(t.pnl_dolares for t in todos_trades if not t.ganadora))

    # Win rate por patrón (todos los activos)
    por_patron: dict = {}
    patrones = set(t.patron for t in todos_trades)
    for p in patrones:
        trades_p = [t for t in todos_trades if t.patron == p]
        gan_p    = sum(1 for t in trades_p if t.ganadora)
        pnl_p    = sum(t.pnl_dolares for t in trades_p)
        r_medio  = sum(t.pnl_r for t in trades_p) / len(trades_p)
        por_patron[p] = {
            "trades":    len(trades_p),
            "win_rate":  round(gan_p / len(trades_p), 3),
            "pnl":       round(pnl_p, 2),
            "r_medio":   round(r_medio, 2),
            "pct_total": round(len(trades_p) / total, 3),
        }

    # Mejor y peor activo
    mejor = max(resultados.items(), key=lambda x: x[1].pnl_total, default=(None, None))
    peor  = min(resultados.items(), key=lambda x: x[1].pnl_total, default=(None, None))

    return {
        "total_trades":     total,
        "ganadoras":        ganadoras,
        "perdedoras":       total - ganadoras,
        "win_rate_global":  round(ganadoras / total, 3),
        "pnl_total":        round(pnl_total, 2),
        "profit_factor":    round(ganancias / perdidas, 2) if perdidas > 0 else 0,
        "r_medio_global":   round(sum(t.pnl_r for t in todos_trades) / total, 2),
        "activos_analizados": len(resultados),
        "por_patron":       por_patron,
        "mejor_activo":     {"symbol": mejor[0], "pnl": round(mejor[1].pnl_total, 2)} if mejor[0] else {},
        "peor_activo":      {"symbol": peor[0],  "pnl": round(peor[1].pnl_total, 2)}  if peor[0]  else {},
    }


def trades_a_dataframe(trades: List[Trade]) -> pd.DataFrame:
    """Convierte lista de trades a DataFrame para análisis."""
    if not trades:
        return pd.DataFrame()

    rows = [{
        "symbol":          t.symbol,
        "patron":          t.patron,
        "direccion":       t.direccion,
        "fecha_entrada":   t.fecha_entrada,
        "fecha_salida":    t.fecha_salida,
        "precio_entrada":  t.precio_entrada,
        "precio_salida":   t.precio_salida,
        "stop_loss":       t.stop_loss,
        "take_profit":     t.take_profit,
        "volumen":         t.volumen,
        "pnl_dolares":     t.pnl_dolares,
        "pnl_r":           t.pnl_r,
        "ganadora":        t.ganadora,
        "motivo_salida":   t.motivo_salida,
        "dias_posicion":   t.dias_en_posicion,
    } for t in trades]

    df = pd.DataFrame(rows)
    df["fecha_entrada"] = pd.to_datetime(df["fecha_entrada"])
    df["fecha_salida"]  = pd.to_datetime(df["fecha_salida"])
    return df


def imprimir_reporte(resultado: ResultadoBacktest) -> None:
    """Imprime reporte completo en consola."""
    r = resultado
    sep = "═" * 60

    print(f"\n{sep}")
    print(f"  BACKTEST: {r.symbol}")
    print(sep)
    print(f"  Total trades:    {r.total_trades:>6}")
    print(f"  Ganadoras:       {r.ganadoras:>6}  ({r.win_rate:.1%})")
    print(f"  Perdedoras:      {r.perdedoras:>6}")
    print(f"  P&L total:       ${r.pnl_total:>10,.2f}")
    print(f"  P&L medio/trade: ${r.pnl_medio:>10,.2f}")
    print(f"  R medio:         {r.r_medio:>6.2f}R")
    print(f"  Profit Factor:   {r.profit_factor:>6.2f}")
    print(f"  Sharpe Ratio:    {r.sharpe_ratio:>6.2f}")
    print(f"  Max Drawdown:    {r.max_drawdown_pct:>6.1%}")
    print(f"  Racha perdedoras:{r.racha_perdedoras:>6}")

    if r.por_patron:
        print(f"\n  {'PATRÓN':<12} {'TRADES':>7} {'WIN%':>7} {'PNL':>12} {'R MEDIO':>8}")
        print("  " + "─" * 50)
        for patron, m in sorted(r.por_patron.items(), key=lambda x: -x[1]["pnl"]):
            print(
                f"  {patron:<12} {m['trades']:>7} "
                f"{m['win_rate']:>6.1%} "
                f"${m['pnl']:>10,.0f} "
                f"{m['r_medio']:>7.2f}R"
            )

    print(sep + "\n")
