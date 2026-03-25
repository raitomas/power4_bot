"""
power4_bot/core/data_fetcher.py
================================================
Descarga datos OHLC desde MT5 en timeframes
semanal y diario. Gestiona caché local en
Parquet para evitar recargas innecesarias.
================================================
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_DISPONIBLE = True
except ImportError:
    mt5 = None
    MT5_DISPONIBLE = False

# ── Mapeo de timeframes legibles → constantes MT5 ──────────────
TIMEFRAMES = {
    "D1": mt5.TIMEFRAME_D1 if MT5_DISPONIBLE else 1440,
    "W1": mt5.TIMEFRAME_W1 if MT5_DISPONIBLE else 10080,
    "H4": mt5.TIMEFRAME_H4 if MT5_DISPONIBLE else 240,
    "H1": mt5.TIMEFRAME_H1 if MT5_DISPONIBLE else 60,
}

# ── Columnas estándar del DataFrame ────────────────────────────
COLUMNAS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]


def _ruta_cache(symbol: str, timeframe: str, cache_dir: str) -> str:
    """Genera la ruta del archivo Parquet de caché."""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{symbol}_{timeframe}.parquet")


def _datos_simulados(symbol: str, n_barras: int) -> pd.DataFrame:
    """
    Genera datos OHLC sintéticos para desarrollo sin MT5.
    Simula una tendencia alcista con retrocesos realistas.
    """
    import numpy as np
    np.random.seed(hash(symbol) % 2**31)

    fechas = pd.bdate_range(end=pd.Timestamp.now(), periods=n_barras)
    precio = 100.0
    rows = []

    for fecha in fechas:
        rango = precio * np.random.uniform(0.008, 0.025)
        drift = precio * np.random.uniform(-0.005, 0.008)
        open_ = precio
        close = precio + drift
        high  = max(open_, close) + rango * np.random.uniform(0.1, 0.5)
        low   = min(open_, close) - rango * np.random.uniform(0.1, 0.5)
        rows.append({
            "time":        fecha,
            "open":        round(open_, 4),
            "high":        round(high, 4),
            "low":         round(low, 4),
            "close":       round(close, 4),
            "tick_volume": int(np.random.uniform(5000, 50000)),
            "spread":      2,
            "real_volume": 0,
        })
        precio = close

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df


def descargar_ohlc(
    symbol: str,
    timeframe: str = "D1",
    n_barras: int = 300,
    cache_dir: str = "data/",
    usar_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Descarga barras OHLC para un símbolo desde MT5.
    Si hay caché reciente (< 1 día), la usa directamente.

    Args:
        symbol:     Símbolo del activo, ej. "AAPL", "EURUSD"
        timeframe:  "D1" (diario) | "W1" (semanal) | "H4" | "H1"
        n_barras:   Número de barras a descargar
        cache_dir:  Carpeta donde guardar el caché Parquet
        usar_cache: Si True, intenta leer caché antes de descargar

    Returns:
        DataFrame con columnas [open, high, low, close, tick_volume]
        indexado por 'time', o None si falla.
    """
    ruta = _ruta_cache(symbol, timeframe, cache_dir)

    # ── 1. Intentar leer caché ──────────────────────────────────
    if usar_cache and os.path.exists(ruta):
        try:
            df_cache = pd.read_parquet(ruta)
            ultima = df_cache.index[-1]
            ahora  = pd.Timestamp.now()
            horas_diferencia = (ahora - ultima).total_seconds() / 3600

            # Caché válida si tiene menos de 24h para D1, 7 días para W1
            limite_horas = 168 if timeframe == "W1" else 24
            if horas_diferencia < limite_horas:
                logger.debug(
                    f"Cache hit: {symbol} {timeframe} "
                    f"({len(df_cache)} barras, última: {ultima.date()})"
                )
                return df_cache
        except Exception as e:
            logger.warning(f"Error leyendo caché {ruta}: {e}")

    # ── 2. Descargar desde MT5 ──────────────────────────────────
    if not MT5_DISPONIBLE:
        logger.info(f"[SIMULACIÓN] Generando datos para {symbol} {timeframe}")
        df = _datos_simulados(symbol, n_barras)
        df.to_parquet(ruta)
        return df

    tf_mt5 = TIMEFRAMES.get(timeframe)
    if tf_mt5 is None:
        logger.error(f"Timeframe no reconocido: {timeframe}")
        return None

    rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, n_barras)

    if rates is None or len(rates) == 0:
        logger.error(
            f"No se pudieron obtener datos para {symbol} {timeframe}. "
            f"Error MT5: {mt5.last_error()}"
        )
        return None

    # ── 3. Construir DataFrame ──────────────────────────────────
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time"] = df["time"].dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    # Mantener solo columnas relevantes
    cols_disponibles = [c for c in ["open", "high", "low", "close", "tick_volume", "spread"] if c in df.columns]
    df = df[cols_disponibles]

    # Asegurar tipos numéricos
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    logger.info(
        f"Descargado: {symbol} {timeframe} — "
        f"{len(df)} barras | "
        f"{df.index[0].date()} → {df.index[-1].date()}"
    )

    # ── 4. Guardar caché ────────────────────────────────────────
    try:
        df.to_parquet(ruta)
        logger.debug(f"Caché guardada: {ruta}")
    except Exception as e:
        logger.warning(f"No se pudo guardar caché {ruta}: {e}")

    return df


def actualizar_incremental(
    symbol: str,
    timeframe: str = "D1",
    cache_dir: str = "data/",
) -> Optional[pd.DataFrame]:
    """
    Actualización incremental: solo descarga las barras nuevas
    desde la última fecha en caché, y las fusiona.
    Más eficiente que descargar 300 barras cada vez.
    """
    ruta = _ruta_cache(symbol, timeframe, cache_dir)

    if not os.path.exists(ruta):
        # No hay caché: descarga completa
        return descargar_ohlc(symbol, timeframe, cache_dir=cache_dir, usar_cache=False)

    try:
        df_existente = pd.read_parquet(ruta)
    except Exception as e:
        logger.warning(f"Error leyendo caché para actualización: {e}")
        return descargar_ohlc(symbol, timeframe, cache_dir=cache_dir, usar_cache=False)

    ultima_fecha = df_existente.index[-1]
    n_barras_nuevas = 10  # Solo las últimas 10 barras para ser eficiente

    if not MT5_DISPONIBLE:
        return df_existente  # En simulación no hay datos nuevos reales

    tf_mt5 = TIMEFRAMES.get(timeframe)
    rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, n_barras_nuevas)

    if rates is None or len(rates) == 0:
        return df_existente

    df_nuevo = pd.DataFrame(rates)
    df_nuevo["time"] = pd.to_datetime(df_nuevo["time"], unit="s", utc=True)
    df_nuevo["time"] = df_nuevo["time"].dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
    df_nuevo.set_index("time", inplace=True)

    # Filtrar solo barras posteriores a la última en caché
    df_nuevo = df_nuevo[df_nuevo.index > ultima_fecha]

    if len(df_nuevo) == 0:
        logger.debug(f"{symbol} {timeframe}: sin barras nuevas.")
        return df_existente

    # Fusionar y guardar
    cols = [c for c in ["open", "high", "low", "close", "tick_volume"] if c in df_nuevo.columns]
    df_nuevo = df_nuevo[cols]
    df_fusionado = pd.concat([df_existente, df_nuevo])
    df_fusionado = df_fusionado[~df_fusionado.index.duplicated(keep="last")]
    df_fusionado.sort_index(inplace=True)
    df_fusionado.to_parquet(ruta)

    logger.info(
        f"Actualizado: {symbol} {timeframe} — "
        f"+{len(df_nuevo)} barras nuevas | total: {len(df_fusionado)}"
    )
    return df_fusionado
