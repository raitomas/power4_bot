"""
power4_bot/tests/test_fase7_8_backtest.py
================================================
Tests para Fase 8 (Backtesting + Métricas).
El dashboard (Fase 7) se valida manualmente con
streamlit run dashboard/app.py
================================================
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.indicators import calcular_indicadores
from backtesting.engine import BacktestEngine, ResultadoBacktest, Trade
from backtesting.metrics import resumen_global, trades_a_dataframe


# ══════════════════════════════════════════════════════════════════
#  FIXTURES
# ══════════════════════════════════════════════════════════════════

def _df_ohlc(n=350, seed=42, drift=0.004):
    """DataFrame OHLC sin indicadores."""
    np.random.seed(seed)
    fechas = pd.bdate_range(end="2026-01-01", periods=n)
    precio = 100.0
    rows   = []
    for _ in fechas:
        d = precio * np.random.uniform(drift * 0.3, drift * 1.7)
        o = precio
        c = precio + d
        h = max(o, c) * np.random.uniform(1.001, 1.010)
        l = min(o, c) * np.random.uniform(0.990, 0.999)
        rows.append({"open": o, "high": h, "low": l, "close": c})
        precio = c
    return pd.DataFrame(rows, index=fechas)


def _trade(ganadora=True, pnl=500.0, patron="PC1"):
    t = Trade(
        symbol        = "AAPL",
        patron        = patron,
        direccion     = "LONG",
        fecha_entrada = date(2025, 1, 10),
        fecha_salida  = date(2025, 1, 20),
        precio_entrada= 100.0,
        precio_salida = 106.0 if ganadora else 93.0,
        stop_loss     = 93.0,
        take_profit   = 114.0,
        volumen       = 71.0,
        pnl_dolares   = pnl if ganadora else -abs(pnl),
        pnl_r         = 2.5 if ganadora else -1.0,
        ganadora      = ganadora,
        motivo_salida = "TP" if ganadora else "SL",
        dias_en_posicion = 10,
    )
    return t


# ══════════════════════════════════════════════════════════════════
#  TESTS ResultadoBacktest.calcular()
# ══════════════════════════════════════════════════════════════════

class TestResultadoBacktest:

    def test_calcular_sin_trades(self):
        r = ResultadoBacktest()
        r.calcular()
        assert r.total_trades == 0

    def test_win_rate_correcto(self):
        r = ResultadoBacktest()
        r.trades = [_trade(True), _trade(True), _trade(False)]
        r.calcular()
        assert abs(r.win_rate - 2/3) < 0.001

    def test_profit_factor_correcto(self):
        r = ResultadoBacktest()
        r.trades = [
            _trade(True,  pnl=600),
            _trade(True,  pnl=400),
            _trade(False, pnl=500),
        ]
        r.calcular()
        # PF = 1000 / 500 = 2.0
        assert abs(r.profit_factor - 2.0) < 0.01

    def test_equity_curve_longitud(self):
        r = ResultadoBacktest()
        r.trades = [_trade(True) for _ in range(5)]
        r.calcular(capital_inicial=100_000)
        assert len(r.equity_curve) == 6  # inicial + 5 trades

    def test_equity_curve_primer_valor_es_capital(self):
        r = ResultadoBacktest()
        r.trades = [_trade(True, pnl=500)]
        r.calcular(capital_inicial=100_000)
        assert r.equity_curve[0] == 100_000

    def test_pnl_total_suma_correcta(self):
        r = ResultadoBacktest()
        r.trades = [_trade(True, 500), _trade(False, 300), _trade(True, 200)]
        r.calcular()
        assert abs(r.pnl_total - (500 - 300 + 200)) < 0.01

    def test_max_drawdown_no_negativo(self):
        r = ResultadoBacktest()
        r.trades = [_trade(True, 500), _trade(False, 800), _trade(True, 300)]
        r.calcular(100_000)
        assert r.max_drawdown_pct >= 0

    def test_racha_perdedoras_correcta(self):
        r = ResultadoBacktest()
        r.trades = [
            _trade(True),
            _trade(False), _trade(False), _trade(False),
            _trade(True),
            _trade(False),
        ]
        r.calcular()
        assert r.racha_perdedoras == 3

    def test_por_patron_agrupado(self):
        r = ResultadoBacktest()
        r.trades = [
            _trade(True,  patron="PC1"),
            _trade(False, patron="PC1"),
            _trade(True,  patron="PV1"),
        ]
        r.calcular()
        assert "PC1" in r.por_patron
        assert "PV1" in r.por_patron
        assert r.por_patron["PC1"]["trades"] == 2
        assert r.por_patron["PV1"]["trades"] == 1

    def test_win_rate_por_patron(self):
        r = ResultadoBacktest()
        r.trades = [
            _trade(True,  patron="PC1"),
            _trade(True,  patron="PC1"),
            _trade(False, patron="PC1"),
        ]
        r.calcular()
        assert abs(r.por_patron["PC1"]["win_rate"] - 2/3) < 0.001


# ══════════════════════════════════════════════════════════════════
#  TESTS BacktestEngine
# ══════════════════════════════════════════════════════════════════

class TestBacktestEngine:

    @pytest.fixture
    def engine(self):
        return BacktestEngine(capital=100_000, riesgo_pct=0.005)

    @pytest.fixture
    def df_alcista(self):
        return _df_ohlc(n=350, drift=0.005)

    def test_ejecutar_retorna_resultado(self, engine, df_alcista):
        df_w1 = _df_ohlc(n=100, drift=0.005)
        r = engine.ejecutar("TEST", df_alcista, df_w1)
        assert isinstance(r, ResultadoBacktest)

    def test_ejecutar_insufficiente_devuelve_vacio(self, engine):
        df_corto = _df_ohlc(n=100)   # Muy corto
        df_w1    = _df_ohlc(n=30)
        r = engine.ejecutar("TEST", df_corto, df_w1)
        assert r.total_trades == 0

    def test_trades_son_trade(self, engine, df_alcista):
        df_w1 = _df_ohlc(n=100, drift=0.005)
        r     = engine.ejecutar("TEST", df_alcista, df_w1)
        for t in r.trades:
            assert isinstance(t, Trade)

    def test_precio_salida_siempre_positivo(self, engine, df_alcista):
        df_w1 = _df_ohlc(n=100, drift=0.005)
        r     = engine.ejecutar("TEST", df_alcista, df_w1)
        for t in r.trades:
            assert t.precio_salida > 0

    def test_fecha_salida_posterior_a_entrada(self, engine, df_alcista):
        df_w1 = _df_ohlc(n=100, drift=0.005)
        r     = engine.ejecutar("TEST", df_alcista, df_w1)
        for t in r.trades:
            if t.fecha_salida and t.fecha_entrada:
                assert t.fecha_salida >= t.fecha_entrada

    def test_motivo_salida_valido(self, engine, df_alcista):
        df_w1   = _df_ohlc(n=100, drift=0.005)
        r       = engine.ejecutar("TEST", df_alcista, df_w1)
        motivos = {"TP", "SL", "TRAILING", "FIN_DATOS"}
        for t in r.trades:
            assert t.motivo_salida in motivos

    def test_evaluar_cierre_tp_long(self, engine):
        """Verifica que el cierre por TP funciona para LONG."""
        pos = Trade(
            symbol="TEST", patron="PC1", direccion="LONG",
            fecha_entrada=date(2025, 1, 1), precio_entrada=100.0,
            stop_loss=93.0, take_profit=115.0, volumen=10.0,
        )
        # Vela que toca TP
        vela = pd.Series({"open": 113.0, "high": 116.0, "low": 112.0, "close": 115.5})
        resultado = engine._evaluar_cierre(pos, vela, pd.Timestamp("2025-01-15"))
        assert resultado is not None
        assert resultado.motivo_salida == "TP"

    def test_evaluar_cierre_sl_long(self, engine):
        """Verifica que el cierre por SL funciona para LONG."""
        pos = Trade(
            symbol="TEST", patron="PC1", direccion="LONG",
            fecha_entrada=date(2025, 1, 1), precio_entrada=100.0,
            stop_loss=93.0, take_profit=115.0, volumen=10.0,
        )
        # Vela que toca SL
        vela = pd.Series({"open": 94.0, "high": 95.0, "low": 92.0, "close": 93.5})
        resultado = engine._evaluar_cierre(pos, vela, pd.Timestamp("2025-01-10"))
        assert resultado is not None
        assert resultado.motivo_salida == "SL"

    def test_evaluar_cierre_ninguno(self, engine):
        """Vela que no toca ni TP ni SL → None."""
        pos = Trade(
            symbol="TEST", patron="PC1", direccion="LONG",
            fecha_entrada=date(2025, 1, 1), precio_entrada=100.0,
            stop_loss=93.0, take_profit=115.0, volumen=10.0,
        )
        vela = pd.Series({"open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0})
        resultado = engine._evaluar_cierre(pos, vela, pd.Timestamp("2025-01-05"))
        assert resultado is None


# ══════════════════════════════════════════════════════════════════
#  TESTS MÉTRICAS
# ══════════════════════════════════════════════════════════════════

class TestMetricas:

    def test_trades_a_dataframe_vacio(self):
        df = trades_a_dataframe([])
        assert df.empty

    def test_trades_a_dataframe_columnas(self):
        df = trades_a_dataframe([_trade()])
        cols_esperadas = ["symbol","patron","direccion","pnl_dolares","ganadora"]
        for c in cols_esperadas:
            assert c in df.columns

    def test_resumen_global_vacio(self):
        r = resumen_global({})
        assert r == {}

    def test_resumen_global_con_datos(self):
        rb = ResultadoBacktest(symbol="AAPL")
        rb.trades = [_trade(True, 500), _trade(False, 300)]
        rb.calcular()
        resumen = resumen_global({"AAPL": rb})
        assert "total_trades"    in resumen
        assert "win_rate_global" in resumen
        assert "profit_factor"   in resumen
        assert resumen["total_trades"] == 2

    def test_resumen_global_por_patron(self):
        rb = ResultadoBacktest(symbol="AAPL")
        rb.trades = [_trade(True, patron="PC1"), _trade(False, patron="PV1")]
        rb.calcular()
        resumen = resumen_global({"AAPL": rb})
        assert "por_patron" in resumen
        assert "PC1" in resumen["por_patron"]
