"""
power4_bot/tests/test_fase1.py
================================================
Tests unitarios para Fase 1.
Funcionan sin MT5 instalado (modo simulación).
Ejecutar: pytest tests/test_fase1.py -v
================================================
"""

import sys
import os
import pytest
import pandas as pd

# Añadir raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_fetcher import descargar_ohlc, _datos_simulados
from core.symbols import _cargar_watchlist


# ── Tests de datos simulados ────────────────────────────────────

class TestDatosSimulados:

    def test_genera_dataframe_correcto(self):
        df = _datos_simulados("AAPL", 300)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 299  # bdate_range puede dar 299 o 300 según día de la semana

    def test_columnas_ohlc_presentes(self):
        df = _datos_simulados("AAPL", 100)
        for col in ["open", "high", "low", "close"]:
            assert col in df.columns, f"Columna {col} faltante"

    def test_high_siempre_mayor_que_low(self):
        df = _datos_simulados("TSLA", 200)
        assert (df["high"] >= df["low"]).all(), "Hay velas donde high < low"

    def test_high_mayor_que_open_y_close(self):
        df = _datos_simulados("NVDA", 200)
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()

    def test_low_menor_que_open_y_close(self):
        df = _datos_simulados("META", 200)
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_indice_es_datetime(self):
        df = _datos_simulados("SPY", 50)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_indice_ordenado_ascendente(self):
        df = _datos_simulados("EURUSD", 100)
        assert df.index.is_monotonic_increasing

    def test_precios_positivos(self):
        df = _datos_simulados("BTCUSD", 100)
        for col in ["open", "high", "low", "close"]:
            assert (df[col] > 0).all(), f"Precios negativos en {col}"


# ── Tests de descarga OHLC ──────────────────────────────────────

class TestDescargarOHLC:

    def test_descarga_d1_devuelve_dataframe(self, tmp_path):
        df = descargar_ohlc("AAPL", "D1", n_barras=300, cache_dir=str(tmp_path))
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 200  # Al menos 200 barras para SMA200

    def test_descarga_w1_devuelve_dataframe(self, tmp_path):
        df = descargar_ohlc("AAPL", "W1", n_barras=80, cache_dir=str(tmp_path))
        assert df is not None
        assert len(df) >= 50

    def test_cache_se_crea(self, tmp_path):
        descargar_ohlc("TSLA", "D1", n_barras=100, cache_dir=str(tmp_path))
        archivos = list(tmp_path.glob("*.parquet"))
        assert len(archivos) == 1, "El archivo de caché no se creó"

    def test_cache_hit_devuelve_mismos_datos(self, tmp_path):
        df1 = descargar_ohlc("NVDA", "D1", n_barras=100, cache_dir=str(tmp_path), usar_cache=False)
        df2 = descargar_ohlc("NVDA", "D1", n_barras=100, cache_dir=str(tmp_path), usar_cache=True)
        assert len(df1) == len(df2)

    def test_suficientes_barras_para_sma200(self, tmp_path):
        """La SMA200 necesita mínimo 200 barras para calcular."""
        df = descargar_ohlc("SPY", "D1", n_barras=300, cache_dir=str(tmp_path))
        assert len(df) >= 200, (
            f"Solo {len(df)} barras. La SMA200 requiere al menos 200."
        )


# ── Tests de watchlist ──────────────────────────────────────────

class TestWatchlist:

    def test_carga_watchlist_yaml(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "config", "watchlist.yaml")
        simbolos = _cargar_watchlist(path)
        assert len(simbolos) > 0

    def test_watchlist_tiene_campos_obligatorios(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "config", "watchlist.yaml")
        simbolos = _cargar_watchlist(path)
        for s in simbolos:
            assert "symbol" in s,   f"Falta 'symbol' en: {s}"
            assert "name" in s,     f"Falta 'name' en: {s}"
            assert "category" in s, f"Falta 'category' en: {s}"

    def test_watchlist_contiene_categorias_esperadas(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "config", "watchlist.yaml")
        simbolos = _cargar_watchlist(path)
        tipos = {s["tipo"] for s in simbolos}
        assert "acciones_usa" in tipos
        assert "forex" in tipos
        assert "crypto" in tipos

    def test_simbolos_sin_espacios(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "config", "watchlist.yaml")
        simbolos = _cargar_watchlist(path)
        for s in simbolos:
            assert " " not in s["symbol"], (
                f"El símbolo '{s['symbol']}' contiene espacios"
            )
