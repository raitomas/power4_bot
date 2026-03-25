"""
power4_bot/tests/test_fase3_classifier.py
================================================
Tests unitarios para el clasificador de etapas.
Ejecutar: pytest tests/test_fase3_classifier.py -v
================================================
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.indicators import calcular_indicadores
from engine.stage_classifier import (
    Etapa,
    ResultadoEtapa,
    ResultadoAlineamiento,
    clasificar_etapa,
    verificar_alineamiento,
    analizar_watchlist,
)


# ══════════════════════════════════════════════════════════════════
#  GENERADORES DE DATOS CONTROLADOS
# ══════════════════════════════════════════════════════════════════

def _df_tendencia_alcista(n: int = 300) -> pd.DataFrame:
    """
    Genera un DataFrame con tendencia alcista clara:
    precio sube ~0.5% cada vela, SMAs apuntan arriba,
    95%+ de cierres sobre SMA20, mínimos crecientes.
    """
    np.random.seed(42)
    fechas = pd.bdate_range(end="2026-01-01", periods=n)
    precio = 100.0
    rows = []
    for _ in fechas:
        drift = precio * np.random.uniform(0.003, 0.008)   # drift alcista forzado
        ruido = precio * np.random.uniform(-0.002, 0.002)
        open_ = precio
        close = precio + drift + ruido
        high  = max(open_, close) * np.random.uniform(1.001, 1.008)
        low   = min(open_, close) * np.random.uniform(0.992, 0.999)
        rows.append({"open": open_, "high": high, "low": low, "close": close})
        precio = close
    df = pd.DataFrame(rows, index=fechas)
    return calcular_indicadores(df)


def _df_tendencia_bajista(n: int = 300) -> pd.DataFrame:
    """Genera un DataFrame con tendencia bajista clara."""
    np.random.seed(99)
    fechas = pd.bdate_range(end="2026-01-01", periods=n)
    precio = 100.0
    rows = []
    for _ in fechas:
        drift = precio * np.random.uniform(-0.008, -0.003)  # drift bajista
        ruido = precio * np.random.uniform(-0.002, 0.002)
        open_ = precio
        close = precio + drift + ruido
        close = max(close, 1.0)  # evitar precios negativos
        high  = max(open_, close) * np.random.uniform(1.001, 1.006)
        low   = min(open_, close) * np.random.uniform(0.994, 0.999)
        rows.append({"open": open_, "high": high, "low": low, "close": close})
        precio = close
    df = pd.DataFrame(rows, index=fechas)
    return calcular_indicadores(df)


def _df_rango_lateral(n: int = 300) -> pd.DataFrame:
    """
    Genera un DataFrame con rango lateral:
    precio oscila alrededor de 100 cruzando SMA20 repetidamente.
    """
    np.random.seed(7)
    fechas = pd.bdate_range(end="2026-01-01", periods=n)
    precio = 100.0
    rows = []
    for i, _ in enumerate(fechas):
        # Oscilación sinusoidal para forzar cruces
        drift = 0.5 * np.sin(i * 0.3) * 0.01 * precio
        ruido = precio * np.random.uniform(-0.003, 0.003)
        open_ = precio
        close = 100.0 + drift + ruido   # anclar cerca de 100
        high  = max(open_, close) * np.random.uniform(1.001, 1.006)
        low   = min(open_, close) * np.random.uniform(0.994, 0.999)
        rows.append({"open": open_, "high": high, "low": low, "close": close})
        precio = close
    df = pd.DataFrame(rows, index=fechas)
    return calcular_indicadores(df)


# ══════════════════════════════════════════════════════════════════
#  TESTS DEL CLASIFICADOR
# ══════════════════════════════════════════════════════════════════

class TestClasificarEtapa:

    def test_retorna_resultado_etapa(self):
        df = _df_tendencia_alcista()
        r = clasificar_etapa(df)
        assert isinstance(r, ResultadoEtapa)

    def test_etapa_alcista_detectada(self):
        """Un activo con drift alcista fuerte debe clasificarse como Etapa 2."""
        df = _df_tendencia_alcista()
        r = clasificar_etapa(df)
        assert r.etapa == Etapa.ALCISTA, (
            f"Se esperaba Etapa 2 (ALCISTA), se obtuvo {r.etapa.name}. "
            f"Razón: {r.razon}"
        )

    def test_etapa_bajista_detectada(self):
        """Un activo con drift bajista fuerte debe clasificarse como Etapa 4."""
        df = _df_tendencia_bajista()
        r = clasificar_etapa(df)
        assert r.etapa == Etapa.BAJISTA, (
            f"Se esperaba Etapa 4 (BAJISTA), se obtuvo {r.etapa.name}. "
            f"Razón: {r.razon}"
        )

    def test_etapa_lateral_detectada(self):
        """Un activo en rango debe ser Etapa 1 o 3 (no 2 ni 4)."""
        df = _df_rango_lateral()
        r = clasificar_etapa(df)
        assert r.etapa in (Etapa.ACUMULACION, Etapa.DISTRIBUCION), (
            f"Se esperaba E1 o E3, se obtuvo {r.etapa.name}. Razón: {r.razon}"
        )

    def test_resultado_tiene_razon(self):
        df = _df_tendencia_alcista()
        r = clasificar_etapa(df)
        assert len(r.razon) > 0, "El resultado debe incluir una razón textual"

    def test_sma20_slope_poblado(self):
        df = _df_tendencia_alcista()
        r = clasificar_etapa(df)
        assert isinstance(r.sma20_slope, float)

    def test_pct_sobre_sma20_entre_0_y_1(self):
        df = _df_tendencia_alcista()
        r = clasificar_etapa(df)
        assert 0.0 <= r.pct_sobre_sma20 <= 1.0

    def test_cruces_sma20_es_entero_no_negativo(self):
        df = _df_rango_lateral()
        r = clasificar_etapa(df)
        assert isinstance(r.cruces_sma20, int)
        assert r.cruces_sma20 >= 0

    def test_df_insuficiente_devuelve_desconocida(self):
        """Con pocas barras no se puede clasificar."""
        df = _df_tendencia_alcista(50)   # Solo 50 barras, insuficiente para SMA200
        # Forzamos un df pequeño sin indicadores
        df_pequeno = df.head(10)[["open","high","low","close"]]
        r = clasificar_etapa(df_pequeno)
        assert r.etapa == Etapa.DESCONOCIDA

    def test_etapa_alcista_tiene_slope_positivo(self):
        df = _df_tendencia_alcista()
        r = clasificar_etapa(df)
        if r.etapa == Etapa.ALCISTA:
            assert r.sma20_slope > 0

    def test_etapa_bajista_tiene_slope_negativo(self):
        df = _df_tendencia_bajista()
        r = clasificar_etapa(df)
        if r.etapa == Etapa.BAJISTA:
            assert r.sma20_slope < 0

    def test_ultimo_close_es_el_ultimo_precio(self):
        df = _df_tendencia_alcista()
        r = clasificar_etapa(df)
        expected = float(df.dropna(subset=["sma20"]).iloc[-1]["close"])
        assert abs(r.ultimo_close - expected) < 0.001

    def test_dist_sma20_pct_no_negativa(self):
        df = _df_tendencia_alcista()
        r = clasificar_etapa(df)
        assert r.dist_sma20_pct >= 0


# ══════════════════════════════════════════════════════════════════
#  TESTS DE ALINEAMIENTO
# ══════════════════════════════════════════════════════════════════

class TestVerificarAlineamiento:

    def test_retorna_resultado_alineamiento(self):
        df_alc = _df_tendencia_alcista()
        r = verificar_alineamiento("TEST", df_alc, df_alc)
        assert isinstance(r, ResultadoAlineamiento)

    def test_symbol_en_resultado(self):
        df = _df_tendencia_alcista()
        r = verificar_alineamiento("AAPL", df, df)
        assert r.symbol == "AAPL"

    def test_alineamiento_long_ambas_e2(self):
        """Si W1=E2 y D1=E2 → LONG alineado."""
        df_alc = _df_tendencia_alcista()
        r = verificar_alineamiento("AAPL", df_alc, df_alc)
        # Ambos deben ser alcistas para alinear
        if r.etapa_w1 == Etapa.ALCISTA and r.etapa_d1 == Etapa.ALCISTA:
            assert r.alineado  == True
            assert r.direccion == "LONG"

    def test_alineamiento_short_ambas_e4(self):
        """Si W1=E4 y D1=E4 → SHORT alineado."""
        df_baj = _df_tendencia_bajista()
        r = verificar_alineamiento("TSLA", df_baj, df_baj)
        if r.etapa_w1 == Etapa.BAJISTA and r.etapa_d1 == Etapa.BAJISTA:
            assert r.alineado  == True
            assert r.direccion == "SHORT"

    def test_no_alineamiento_etapas_distintas(self):
        """Si W1=E2 y D1=E4 → no alineado."""
        df_alc = _df_tendencia_alcista()
        df_baj = _df_tendencia_bajista()
        r = verificar_alineamiento("MIX", df_alc, df_baj)
        # Solo alineado si ambas etapas coinciden en 2 o en 4
        if r.etapa_w1 != r.etapa_d1:
            assert r.alineado == False
            assert r.direccion is None

    def test_operable_requiere_alineado_y_cerca_sma20(self):
        """operable = alineado AND cerca_sma20."""
        df = _df_tendencia_alcista()
        r = verificar_alineamiento("SPY", df, df)
        if r.operable:
            assert r.alineado == True
            assert r.cerca_sma20 == True

    def test_dist_sma20_es_float_no_negativo(self):
        df = _df_tendencia_alcista()
        r = verificar_alineamiento("QQQ", df, df)
        assert isinstance(r.dist_sma20, float)
        assert r.dist_sma20 >= 0

    def test_detalle_w1_y_d1_poblados(self):
        df = _df_tendencia_alcista()
        r = verificar_alineamiento("META", df, df)
        assert r.detalle_w1 is not None
        assert r.detalle_d1 is not None
        assert isinstance(r.detalle_w1, ResultadoEtapa)
        assert isinstance(r.detalle_d1, ResultadoEtapa)


# ══════════════════════════════════════════════════════════════════
#  TESTS DE ANALIZAR WATCHLIST
# ══════════════════════════════════════════════════════════════════

class TestAnalizarWatchlist:

    @pytest.fixture
    def datos_watchlist(self):
        """Watchlist simulada con 4 activos en distintas etapas."""
        return {
            "AAPL": {"W1": _df_tendencia_alcista(), "D1": _df_tendencia_alcista()},
            "TSLA": {"W1": _df_tendencia_bajista(), "D1": _df_tendencia_bajista()},
            "SPY":  {"W1": _df_rango_lateral(),     "D1": _df_rango_lateral()},
            "META": {"W1": _df_tendencia_alcista(), "D1": _df_rango_lateral()},
        }

    def test_retorna_lista(self, datos_watchlist):
        resultado = analizar_watchlist(datos_watchlist)
        assert isinstance(resultado, list)

    def test_longitud_correcta(self, datos_watchlist):
        resultado = analizar_watchlist(datos_watchlist)
        assert len(resultado) == 4

    def test_todos_son_resultado_alineamiento(self, datos_watchlist):
        resultado = analizar_watchlist(datos_watchlist)
        for r in resultado:
            assert isinstance(r, ResultadoAlineamiento)

    def test_operables_aparecen_primero(self, datos_watchlist):
        resultado = analizar_watchlist(datos_watchlist)
        # Los operables deben estar antes que los no operables
        operables_idx = [i for i, r in enumerate(resultado) if r.operable]
        no_operables_idx = [i for i, r in enumerate(resultado) if not r.operable]
        if operables_idx and no_operables_idx:
            assert max(operables_idx) < min(no_operables_idx)

    def test_sin_datos_w1_se_salta(self):
        datos_incompletos = {
            "AAPL": {"D1": _df_tendencia_alcista()},  # falta W1
            "NVDA": {"W1": _df_tendencia_alcista(), "D1": _df_tendencia_alcista()},
        }
        resultado = analizar_watchlist(datos_incompletos)
        symbols = [r.symbol for r in resultado]
        assert "NVDA" in symbols
        assert "AAPL" not in symbols   # saltado por falta de W1

    def test_watchlist_vacia(self):
        resultado = analizar_watchlist({})
        assert resultado == []


# ══════════════════════════════════════════════════════════════════
#  TESTS DE ENUM Y TIPOS
# ══════════════════════════════════════════════════════════════════

class TestEnumEtapa:

    def test_valores_enteros_correctos(self):
        assert int(Etapa.DESCONOCIDA)  == 0
        assert int(Etapa.ACUMULACION)  == 1
        assert int(Etapa.ALCISTA)      == 2
        assert int(Etapa.DISTRIBUCION) == 3
        assert int(Etapa.BAJISTA)      == 4

    def test_comparacion_etapas(self):
        assert Etapa.ALCISTA > Etapa.ACUMULACION
        assert Etapa.BAJISTA > Etapa.DISTRIBUCION

    def test_nombre_etapa(self):
        assert Etapa(2).name == "ALCISTA"
        assert Etapa(4).name == "BAJISTA"
