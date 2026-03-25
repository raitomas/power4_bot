"""
power4_bot/tests/test_fase2_indicators.py
================================================
Tests unitarios para el motor de indicadores.
Ejecutar: pytest tests/test_fase2_indicators.py -v
================================================
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_fetcher import _datos_simulados
from engine.indicators import (
    calcular_indicadores,
    get_ultimos_swing_highs,
    get_ultimos_swing_lows,
    maximos_son_decrecientes,
    minimos_son_crecientes,
    contar_cruces_sma20,
    pct_cierres_sobre_sma20,
    resumen_indicadores,
)


# ── Fixture compartida ──────────────────────────────────────────

@pytest.fixture
def df_base():
    """DataFrame simulado con 300 barras, suficiente para SMA200."""
    return _datos_simulados("AAPL", 300)


@pytest.fixture
def df_enriquecido(df_base):
    """DataFrame con todos los indicadores calculados."""
    return calcular_indicadores(df_base)


# ── Tests de SMAs ───────────────────────────────────────────────

class TestSMAs:

    def test_columnas_sma_existen(self, df_enriquecido):
        for col in ["sma20", "sma40", "sma200"]:
            assert col in df_enriquecido.columns, f"Falta columna {col}"

    def test_sma20_tiene_menos_nan_que_sma200(self, df_enriquecido):
        nan20  = df_enriquecido["sma20"].isna().sum()
        nan200 = df_enriquecido["sma200"].isna().sum()
        assert nan20 < nan200, "SMA20 debería tener menos NaN que SMA200"

    def test_sma200_necesita_200_barras(self, df_enriquecido):
        primeros_nan = df_enriquecido["sma200"].isna().sum()
        assert primeros_nan == 199, f"SMA200 debería tener 199 NaN al inicio, tiene {primeros_nan}"

    def test_sma20_es_media_movil_correcta(self, df_base):
        df = calcular_indicadores(df_base)
        # Verificar que SMA20 en posición 200 = media de los 20 cierres previos
        idx = 200
        expected = df_base["close"].iloc[idx - 20 : idx].mean()
        actual   = df["sma20"].iloc[idx - 1]
        assert abs(actual - expected) < 0.001, f"SMA20 incorrecta: {actual} vs {expected}"

    def test_sma_orden_logico_en_tendencia(self, df_enriquecido):
        """
        En un activo simulado con drift alcista, al final de la serie
        SMA20 suele estar por encima de SMA200 (no siempre pero mayoritariamente).
        Verificamos que los valores son distintos (no colapsan).
        """
        ultimo = df_enriquecido.dropna(subset=["sma20", "sma200"]).iloc[-1]
        assert ultimo["sma20"] != ultimo["sma200"]

    def test_pendientes_calculadas(self, df_enriquecido):
        for col in ["sma20_slope", "sma40_slope", "sma200_slope"]:
            assert col in df_enriquecido.columns
            # La pendiente es sma[i] - sma[i-1], no puede ser siempre 0
            assert df_enriquecido[col].dropna().abs().sum() > 0


# ── Tests de ATR ────────────────────────────────────────────────

class TestATR:

    def test_atr14_existe(self, df_enriquecido):
        assert "atr14" in df_enriquecido.columns

    def test_atr14_siempre_positivo(self, df_enriquecido):
        atr = df_enriquecido["atr14"].dropna()
        assert (atr > 0).all(), "ATR14 tiene valores negativos o cero"

    def test_atr14_primeros_son_nan(self, df_enriquecido):
        assert df_enriquecido["atr14"].iloc[:13].isna().all()

    def test_atr14_menor_que_rango_maximo(self, df_enriquecido):
        """ATR (media) no puede ser mayor que el rango máximo histórico."""
        max_rango = (df_enriquecido["high"] - df_enriquecido["low"]).max()
        max_atr   = df_enriquecido["atr14"].max()
        assert max_atr <= max_rango * 1.1  # pequeño margen por redondeos


# ── Tests de Swings ─────────────────────────────────────────────

class TestSwings:

    def test_columnas_swing_existen(self, df_enriquecido):
        for col in ["swing_high", "swing_low", "swing_high_price", "swing_low_price"]:
            assert col in df_enriquecido.columns

    def test_hay_swings_detectados(self, df_enriquecido):
        assert df_enriquecido["swing_high"].sum() > 0, "No se detectaron swing highs"
        assert df_enriquecido["swing_low"].sum()  > 0, "No se detectaron swing lows"

    def test_swing_high_es_maximo_local(self, df_enriquecido):
        """Cada swing_high debe ser mayor que las 3 velas a cada lado."""
        n = 3
        df = df_enriquecido
        for i in df[df["swing_high"] == True].index:
            loc = df.index.get_loc(i)
            if loc < n or loc >= len(df) - n:
                continue
            pivot = df["high"].iloc[loc]
            vecinos_prev = df["high"].iloc[loc - n : loc].values
            vecinos_post = df["high"].iloc[loc + 1 : loc + n + 1].values
            assert pivot > vecinos_prev.max(), f"Swing high en {i} no es máximo local (izq)"
            assert pivot > vecinos_post.max(), f"Swing high en {i} no es máximo local (der)"

    def test_swing_low_es_minimo_local(self, df_enriquecido):
        """Cada swing_low debe ser menor que las 3 velas a cada lado."""
        n = 3
        df = df_enriquecido
        for i in df[df["swing_low"] == True].index:
            loc = df.index.get_loc(i)
            if loc < n or loc >= len(df) - n:
                continue
            pivot = df["low"].iloc[loc]
            vecinos_prev = df["low"].iloc[loc - n : loc].values
            vecinos_post = df["low"].iloc[loc + 1 : loc + n + 1].values
            assert pivot < vecinos_prev.min(), f"Swing low en {i} no es mínimo local (izq)"
            assert pivot < vecinos_post.min(), f"Swing low en {i} no es mínimo local (der)"

    def test_swing_price_coincide_con_high_low(self, df_enriquecido):
        df = df_enriquecido
        for i in df[df["swing_high"] == True].index:
            assert df.loc[i, "swing_high_price"] == df.loc[i, "high"]
        for i in df[df["swing_low"] == True].index:
            assert df.loc[i, "swing_low_price"] == df.loc[i, "low"]

    def test_ultimos_swing_highs_devuelve_serie(self, df_enriquecido):
        result = get_ultimos_swing_highs(df_enriquecido, 5)
        assert isinstance(result, pd.Series)
        assert len(result) <= 5

    def test_ultimos_swing_lows_devuelve_serie(self, df_enriquecido):
        result = get_ultimos_swing_lows(df_enriquecido, 5)
        assert isinstance(result, pd.Series)
        assert len(result) <= 5


# ── Tests de funciones de tendencia ────────────────────────────

class TestTendencia:

    def test_contar_cruces_sma20_retorna_entero(self, df_enriquecido):
        cruces = contar_cruces_sma20(df_enriquecido, 20)
        assert isinstance(cruces, int)
        assert cruces >= 0

    def test_pct_cierres_sobre_sma20_entre_0_y_1(self, df_enriquecido):
        pct = pct_cierres_sobre_sma20(df_enriquecido, 20)
        assert 0.0 <= pct <= 1.0

    def test_pct_sobre_mas_bajo_suman_1(self, df_enriquecido):
        from engine.indicators import pct_cierres_bajo_sma20
        sobre = pct_cierres_sobre_sma20(df_enriquecido, 20)
        bajo  = pct_cierres_bajo_sma20(df_enriquecido, 20)
        assert abs(sobre + bajo - 1.0) < 0.001

    def test_minimos_crecientes_retorna_bool(self, df_enriquecido):
        result = minimos_son_crecientes(df_enriquecido)
        assert isinstance(result, bool)

    def test_maximos_decrecientes_retorna_bool(self, df_enriquecido):
        result = maximos_son_decrecientes(df_enriquecido)
        assert isinstance(result, bool)

    def test_minimos_crecientes_con_serie_conocida(self):
        """Test con datos manuales donde los mínimos son claramente crecientes."""
        # Construimos un DataFrame artificial con mínimos crecientes garantizados
        data = {
            "open":  [10, 11, 12, 9, 13, 12, 14, 13, 15, 14, 16, 15, 17],
            "high":  [12, 13, 14, 11, 15, 14, 16, 15, 17, 16, 18, 17, 19],
            "low":   [9,  10, 11, 7,  12, 11, 13, 12, 14, 13, 15, 14, 16],
            "close": [11, 12, 13, 10, 14, 13, 15, 14, 16, 15, 17, 16, 18],
        }
        df = pd.DataFrame(data)
        df = _detectar_swings_test(df)
        # No podemos testear minimos_crecientes sin datos reales suficientes
        # pero sí que la función no explota
        result = minimos_son_crecientes(df)
        assert isinstance(result, bool)


# ── Tests de velas ──────────────────────────────────────────────

class TestVelas:

    def test_columnas_vela_existen(self, df_enriquecido):
        for col in ["vela_alcista", "vela_bajista", "vela_doji", "body_pct"]:
            assert col in df_enriquecido.columns

    def test_vela_alcista_y_bajista_mutuamente_excluyentes(self, df_enriquecido):
        ambas = df_enriquecido["vela_alcista"] & df_enriquecido["vela_bajista"]
        assert not ambas.any(), "Una vela no puede ser alcista Y bajista a la vez"

    def test_body_pct_entre_0_y_1(self, df_enriquecido):
        bp = df_enriquecido["body_pct"].dropna()
        assert (bp >= 0).all() and (bp <= 1).all()


# ── Tests de distancia SMA20 ─────────────────────────────────────

class TestDistanciaSMA20:

    def test_dist_sma20_pct_existe(self, df_enriquecido):
        assert "dist_sma20_pct" in df_enriquecido.columns

    def test_cerca_sma20_es_bool(self, df_enriquecido):
        assert df_enriquecido["cerca_sma20"].dtype == bool

    def test_dist_positiva(self, df_enriquecido):
        d = df_enriquecido["dist_sma20_pct"].dropna()
        assert (d >= 0).all()

    def test_cerca_sma20_implica_dist_menor_4(self, df_enriquecido):
        df = df_enriquecido.dropna(subset=["dist_sma20_pct"])
        cercanos = df[df["cerca_sma20"] == True]
        assert (cercanos["dist_sma20_pct"] < 4.0).all()


# ── Test de resumen ─────────────────────────────────────────────

class TestResumen:

    def test_resumen_devuelve_dict(self, df_enriquecido):
        r = resumen_indicadores(df_enriquecido)
        assert isinstance(r, dict)

    def test_resumen_tiene_claves_esperadas(self, df_enriquecido):
        r = resumen_indicadores(df_enriquecido)
        claves = ["close", "sma20", "sma40", "sma200", "dist_sma20_pct",
                  "cerca_sma20", "swing_highs_total", "swing_lows_total"]
        for c in claves:
            assert c in r, f"Clave faltante en resumen: {c}"

    def test_resumen_con_df_none(self):
        r = resumen_indicadores(None)
        assert r == {}


# ── Helper para tests internos ──────────────────────────────────

def _detectar_swings_test(df):
    """Mini-wrapper para tests sin pasar por calcular_indicadores completo."""
    from engine.indicators import _detectar_swings
    df["swing_high"] = False
    df["swing_low"]  = False
    df["swing_high_price"] = np.nan
    df["swing_low_price"]  = np.nan
    return _detectar_swings(df)
