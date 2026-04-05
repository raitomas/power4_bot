"""
backtest_majors_1y.py
================================================
Backtesting 5 años — Universo completo Power 4
Método Power 4

Categorias:
  · Forex Majors    (7 pares)
  · Forex Crosses   (EURGBP, GBPJPY, EURJPY)
  · Indices         (SP500, Nasdaq, Dow, DAX, Nikkei, FTSE, Stoxx50)
  · Metales         (XAUUSD, XAGUSD)
  · Acciones Nasdaq (AAPL, MSFT, NVDA, TSLA, META, AMZN, GOOGL,
                     AVGO, AMD, NFLX)

Parametros:
  Capital inicial:   $100,000
  Riesgo por op:     0.5% ($500)
  Periodo activo:    5 años (~1260 barras D1 activas)
  SL calibrado por categoria de activo

Requiere MetaTrader 5 activo y conectado al broker.
================================================
"""

import sys
import os
import logging
from datetime import datetime

# ── Path del proyecto ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("backtest_power4")

# ── Importaciones del proyecto ────────────────────────────────────
try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 no instalado. Ejecuta: pip install MetaTrader5")
    sys.exit(1)

from backtesting.engine import BacktestEngine
from backtesting.metrics import resumen_global, trades_a_dataframe, imprimir_reporte
from core.data_fetcher import descargar_ohlc

# ══════════════════════════════════════════════════════════════════
#  UNIVERSO DE ACTIVOS (por categoria con parametros propios)
# ══════════════════════════════════════════════════════════════════

CAPITAL    = 100_000.0
RIESGO_PCT = 0.005     # 0.5% = $500 por operacion
MIN_BARRAS = 250       # Barras de calentamiento para indicadores
MIN_RR     = 1.5       # R/R minimo aceptable

# D1: 250 warmup + 1260 activos (~5 años)
N_D1 = 1510
# W1: 350 barras (~7 años para contexto de etapa semanal)
N_W1 = 350

UNIVERSE = {
    # ── Forex Majors ─────────────────────────────────────────────
    # SL tipico: 0.3-1.5% del precio (30-150 pips en EURUSD)
    "FOREX_MAJORS": {
        "label":    "Forex Majors",
        "symbols":  ["EURUSD", "GBPUSD", "USDJPY",
                     "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"],
        "sl_min":   0.003,   # 0.3% min (~30 pips)
        "sl_max":   0.025,   # 2.5% max (~270 pips, mov. excepcional)
        "comision": 0.0001,  # ~0.01% por lado (spread 1-2 pips)
    },
    # ── Forex Crosses solicitados ────────────────────────────────
    # Pares con spread algo mayor, misma logica de distancia
    "FOREX_CROSSES": {
        "label":    "Forex Crosses",
        "symbols":  ["EURGBP", "GBPJPY", "EURJPY"],
        "sl_min":   0.003,
        "sl_max":   0.030,   # Cruces JPY pueden moverse mas
        "comision": 0.00015, # Spread ligeramente mayor
    },
    # ── Indices principales ──────────────────────────────────────
    # SL tipico: 2-6% del nivel (el metodo menciona 6-7% para stocks)
    "INDICES": {
        "label":    "Indices",
        "symbols":  ["SP500", "SP500m", "US30",
                     "DE40", "NI225", "UK100", "STOX50"],
        "sl_min":   0.015,   # 1.5% min
        "sl_max":   0.10,    # 10% max (indices pueden tener SL amplios)
        "comision": 0.0002,  # Spread CFD indices
    },
    # ── Metales ──────────────────────────────────────────────────
    # Oro y Plata: volatilidad intermedia entre forex e indices
    "METALS": {
        "label":    "Metales",
        "symbols":  ["XAUUSD", "XAGUSD"],
        "sl_min":   0.005,   # 0.5% (~10$ en oro a 2000)
        "sl_max":   0.06,    # 6%
        "comision": 0.0002,
    },
    # ── Acciones Nasdaq (Top) ────────────────────────────────────
    # El metodo dice ~6-7% de distancia al stop para acciones
    "NASDAQ_TOP": {
        "label":    "Acciones Nasdaq",
        "symbols":  ["AAPL", "MSFT", "NVDA", "TSLA",
                     "META", "AMZN", "GOOGL",
                     "AVGO", "AMD",  "NFLX"],
        "sl_min":   0.03,    # 3% min
        "sl_max":   0.15,    # 15% max (TSLA/NVDA muy volatiles)
        "comision": 0.0005,  # Comision broker acciones
    },
}

# ══════════════════════════════════════════════════════════════════
#  HELPERS DE FORMATO
# ══════════════════════════════════════════════════════════════════

SEP1 = "=" * 80
SEP2 = "-" * 80


def _pf(pf: float) -> str:
    return f"{pf:.2f}" if pf != float("inf") else "inf"


def imprimir_cabecera(n_activos: int, n_anos: int):
    print(f"\n{SEP1}")
    titulo = f"BACKTESTING METODO POWER 4  |  {n_activos} ACTIVOS  |  {n_anos} ANOS"
    print(f"{titulo:^80}")
    subtit = f"Capital: $100,000  |  Riesgo: 0.5%/op  |  R/R min: 1.5"
    print(f"{subtit:^80}")
    print(f"{'Fecha: ' + datetime.now().strftime('%Y-%m-%d %H:%M'):^80}")
    print(SEP1)


def imprimir_seccion(label: str, resultados: dict):
    """Imprime tabla de resultados para una categoria."""
    # Cabecera de seccion
    print(f"\n  [{label.upper()}]")
    print(f"  {'ACTIVO':<10} {'TRADES':>7} {'WIN%':>7} {'PF':>7} "
          f"{'SHARPE':>7} {'MAX DD':>8} {'R MED':>7} {'P&L ($)':>12}")
    print("  " + "-" * 68)

    for sym, r in resultados.items():
        if r.total_trades == 0:
            print(f"  {sym:<10}  sin señales alineadas en el periodo")
            continue
        signo = "+" if r.pnl_total >= 0 else ""
        print(
            f"  {sym:<10} "
            f"{r.total_trades:>7} "
            f"{r.win_rate:>6.1%} "
            f"{_pf(r.profit_factor):>7} "
            f"{r.sharpe_ratio:>7.2f} "
            f"{r.max_drawdown_pct:>7.1%} "
            f"{r.r_medio:>6.2f}R "
            f"  {signo}${r.pnl_total:>10,.0f}"
        )


def imprimir_resumen_global(resultados_todos: dict):
    """Resumen consolidado de todos los activos."""
    glob = resumen_global(resultados_todos)
    if not glob:
        print("\n  Sin datos para resumen global.\n")
        return

    print(f"\n{SEP1}")
    print(f"{'RESUMEN GLOBAL — TODOS LOS ACTIVOS':^80}")
    print(SEP1)
    print(f"  Activos analizados:    {glob['activos_analizados']}")
    print(f"  Total operaciones:     {glob['total_trades']}")
    ganadas = glob['ganadoras']
    wr      = glob['win_rate_global']
    print(f"  Ganadoras:             {ganadas}  ({wr:.1%})")
    print(f"  Perdedoras:            {glob['perdedoras']}")
    print(f"  P&L total:            ${glob['pnl_total']:,.2f}")
    print(f"  Profit Factor:         {_pf(glob['profit_factor'])}")
    print(f"  R medio global:        {glob['r_medio_global']:.2f}R")
    if glob.get("mejor_activo"):
        m = glob["mejor_activo"]
        print(f"  Mejor activo:          {m['symbol']}  (+${m['pnl']:,.0f})")
    if glob.get("peor_activo"):
        p = glob["peor_activo"]
        print(f"  Peor activo:           {p['symbol']}  (${p['pnl']:,.0f})")

    # Ranking de patrones
    if glob.get("por_patron"):
        print(f"\n  {'PATRON':<16} {'TRADES':>7} {'WIN%':>7} {'R MEDIO':>9} {'P&L ($)':>13}")
        print("  " + "-" * 56)
        for patron, m in sorted(
            glob["por_patron"].items(), key=lambda x: -x[1]["pnl"]
        ):
            signo = "+" if m["pnl"] >= 0 else ""
            print(
                f"  {patron:<16} "
                f"{m['trades']:>7} "
                f"{m['win_rate']:>6.1%} "
                f"{m['r_medio']:>8.2f}R "
                f"  {signo}${m['pnl']:>10,.0f}"
            )

    print(SEP1 + "\n")


# ══════════════════════════════════════════════════════════════════
#  FUNCION PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def main():
    n_activos = sum(len(v["symbols"]) for v in UNIVERSE.values())
    n_anos    = round((N_D1 - MIN_BARRAS) / 252)
    imprimir_cabecera(n_activos, n_anos)

    # ── 1. Inicializar MT5 ─────────────────────────────────────
    print(f"\n  Conectando MetaTrader 5...", end=" ", flush=True)
    if not mt5.initialize():
        print(f"FALLO  ({mt5.last_error()})")
        return
    info = mt5.terminal_info()
    broker = info.name if info else "broker desconocido"
    print(f"OK  ({broker})")
    print(f"  Periodo activo: ~{n_anos} años (~{N_D1 - MIN_BARRAS} barras D1)")
    print(f"  Total activos:  {n_activos}\n")

    todos_resultados  = {}   # {symbol: ResultadoBacktest}
    resultados_por_cat = {}  # {categoria: {symbol: ResultadoBacktest}}

    # ── 2. Iterar por categoria ────────────────────────────────
    for cat_key, cat in UNIVERSE.items():
        label   = cat["label"]
        symbols = cat["symbols"]
        print(f"\n{SEP2}")
        print(f"  {label.upper()}  ({len(symbols)} activos)")
        print(SEP2)

        # Motor calibrado para esta categoria
        engine = BacktestEngine(
            capital         = CAPITAL,
            riesgo_pct      = RIESGO_PCT,
            comision_pct    = cat["comision"],
            min_barras      = MIN_BARRAS,
            dist_sl_min_pct = cat["sl_min"],
            dist_sl_max_pct = cat["sl_max"],
            min_ratio_rr    = MIN_RR,
        )

        cat_resultados = {}

        for sym in symbols:
            print(f"  > {sym:<8}", end=" ", flush=True)

            df_d1 = descargar_ohlc(sym, "D1", n_barras=N_D1)
            df_w1 = descargar_ohlc(sym, "W1", n_barras=N_W1)

            # Validar datos
            if df_d1 is None or len(df_d1) < MIN_BARRAS + 50:
                n_d1 = len(df_d1) if df_d1 is not None else 0
                print(f"ERROR datos D1 insuficientes ({n_d1} barras)")
                continue

            if df_w1 is None or len(df_w1) < 50:
                print(f"ERROR datos W1 insuficientes")
                continue

            # Rango de fechas activo (desde barra warmup)
            f_ini = df_d1.index[MIN_BARRAS].strftime("%Y-%m-%d")
            f_fin = df_d1.index[-1].strftime("%Y-%m-%d")
            n_activo = len(df_d1) - MIN_BARRAS
            print(
                f"D1:{len(df_d1)}  W1:{len(df_w1)}  "
                f"[{f_ini} -> {f_fin}]  activas:{n_activo}",
                end=" ... ",
                flush=True
            )

            try:
                res = engine.ejecutar(sym, df_d1, df_w1)
                cat_resultados[sym]  = res
                todos_resultados[sym] = res

                if res.total_trades > 0:
                    signo = "+" if res.pnl_total >= 0 else ""
                    print(
                        f"{res.total_trades} trades | "
                        f"WR={res.win_rate:.0%} | "
                        f"PF={_pf(res.profit_factor)} | "
                        f"P&L={signo}${res.pnl_total:,.0f}"
                    )
                else:
                    print("0 trades (sin alineamiento)")

            except Exception as e:
                print(f"ERROR: {e}")
                logger.exception(f"Backtest {sym}")

        resultados_por_cat[cat_key] = cat_resultados

    # ── 3. Tablas de resultados por categoria ──────────────────
    print(f"\n\n{SEP1}")
    print(f"{'RESULTADOS POR CATEGORIA':^80}")
    print(SEP1)

    for cat_key, cat in UNIVERSE.items():
        if cat_key in resultados_por_cat:
            imprimir_seccion(cat["label"], resultados_por_cat[cat_key])

    # ── 4. Detalle individual (solo activos con trades) ────────
    activos_con_trades = {s: r for s, r in todos_resultados.items()
                          if r.total_trades > 0}

    if activos_con_trades:
        print(f"\n\n{SEP1}")
        print(f"{'DETALLE COMPLETO (ACTIVOS CON OPERACIONES)':^80}")
        print(SEP1)
        for sym, res in activos_con_trades.items():
            imprimir_reporte(res)

    # ── 5. Resumen global ──────────────────────────────────────
    imprimir_resumen_global(todos_resultados)

    # ── 6. Exportar CSV ────────────────────────────────────────
    todos_trades = []
    for res in todos_resultados.values():
        todos_trades.extend(res.trades)

    if todos_trades:
        try:
            df_trades = trades_a_dataframe(todos_trades)
            ts        = datetime.now().strftime("%Y%m%d_%H%M")
            nombre    = f"backtest_power4_{n_anos}y_{ts}.csv"
            ruta_csv  = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), nombre
            )
            df_trades.to_csv(ruta_csv, index=False)
            print(f"  CSV con {len(df_trades)} operaciones guardado:")
            print(f"  {ruta_csv}")
        except Exception as e:
            print(f"  (CSV no guardado: {e})")
    else:
        print("  Sin operaciones para exportar.")

    mt5.shutdown()
    print(f"\n  Fin del backtesting — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")


if __name__ == "__main__":
    main()
