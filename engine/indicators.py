"""
power4_bot/engine/indicators.py
================================================
Motor de indicadores técnicos del Método Power 4.

Calcula sobre un DataFrame OHLC:
  - SMA 20, 40, 200 + sus pendientes
  - Máximos y Mínimos Relevantes (swing points)
  - Distancia % del precio a SMA 20
  - Flag "cerca_sma20" (Regla Barrio Sésamo)
  - Clasificación de velas (alcista/bajista/doji)
  - ATR(14) para contexto de volatilidad
================================================
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Parámetros por defecto (sobreescribibles desde settings.yaml) ──
SMA_RAPIDA  = 20
SMA_MEDIA   = 40
SMA_LENTA   = 200
ATR_PERIOD  = 14
SWING_VELAS = 3          # Velas a cada lado para confirmar swing
MAX_DIST_SMA20 = 4.0     # % máximo distancia SMA20 (Barrio Sésamo)


# ══════════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def calcular_indicadores(
    df: pd.DataFrame,
    sma_rapida: int = SMA_RAPIDA,
    sma_media:  int = SMA_MEDIA,
    sma_lenta:  int = SMA_LENTA,
    dist_max_pct: float = MAX_DIST_SMA20,
) -> pd.DataFrame:
    """
    Recibe un DataFrame OHLC y devuelve el mismo DataFrame
    enriquecido con todas las columnas de indicadores.

    Columnas añadidas:
        sma20, sma40, sma200
        sma20_slope, sma40_slope, sma200_slope
        atr14
        swing_high, swing_low          (bool: es punto relevante)
        swing_high_price, swing_low_price (precio del swing o NaN)
        dist_sma20_pct                 (% distancia close a SMA20)
        cerca_sma20                    (bool: dist < dist_max_pct)
        vela_alcista, vela_bajista, vela_doji  (bool)
        body_pct                       (tamaño cuerpo / rango total)

    Args:
        df: DataFrame con columnas [open, high, low, close]
        sma_rapida / sma_media / sma_lenta: periodos de las SMAs
        dist_max_pct: umbral "Barrio Sésamo" en porcentaje

    Returns:
        DataFrame enriquecido (copia, no modifica el original)
    """
    if df is None or len(df) < sma_rapida:
        logger.warning(
            f"DataFrame insuficiente ({len(df) if df is not None else 0} barras). "
            f"Se necesitan mínimo {sma_rapida} para SMA{sma_rapida}."
        )
        return df

    if len(df) < sma_lenta:
        logger.debug(
            f"Solo {len(df)} barras — SMA{sma_lenta} no disponible "
            f"(se necesitan {sma_lenta}). SMA{sma_rapida} y SMA{sma_media} sí se calcularán."
        )

    df = df.copy()

    # ── 1. Medias Móviles Simples ─────────────────────────────────
    df = _calcular_smas(df, sma_rapida, sma_media, sma_lenta)

    # ── 2. Pendientes de las SMAs ─────────────────────────────────
    df = _calcular_pendientes(df)

    # ── 3. ATR(14) ────────────────────────────────────────────────
    df = _calcular_atr(df, ATR_PERIOD)

    # ── 4. Swing Highs y Swing Lows (Máx/Mín Relevantes) ─────────
    df = _detectar_swings(df, SWING_VELAS)

    # ── 5. Distancia a SMA20 ──────────────────────────────────────
    df["dist_sma20_pct"] = (
        (df["close"] - df["sma20"]).abs() / df["sma20"] * 100
    ).round(2)
    df["cerca_sma20"] = df["dist_sma20_pct"] < dist_max_pct

    # ── 6. Clasificación de velas ─────────────────────────────────
    df = _clasificar_velas(df)

    logger.debug(
        f"Indicadores calculados: {len(df)} barras | "
        f"SMA{sma_rapida}/{sma_media}/{sma_lenta} | "
        f"Swings altos: {df['swing_high'].sum()} | "
        f"Swings bajos: {df['swing_low'].sum()}"
    )

    return df


# ══════════════════════════════════════════════════════════════════
#  FUNCIONES INTERNAS
# ══════════════════════════════════════════════════════════════════

def _calcular_smas(df: pd.DataFrame, r: int, m: int, l: int) -> pd.DataFrame:
    """Calcula las tres SMAs al cierre."""
    df[f"sma{r}"]  = df["close"].rolling(window=r, min_periods=r).mean().round(4)
    df[f"sma{m}"]  = df["close"].rolling(window=m, min_periods=m).mean().round(4)
    df[f"sma{l}"]  = df["close"].rolling(window=l, min_periods=l).mean().round(4)
    # Alias estándar para el resto de módulos
    df["sma20"]  = df[f"sma{r}"]
    df["sma40"]  = df[f"sma{m}"]
    df["sma200"] = df[f"sma{l}"]
    return df


def _calcular_pendientes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pendiente = SMA[i] - SMA[i-1].
    Positiva → media apunta arriba (alcista)
    Negativa → media apunta abajo (bajista)
    ~0       → media plana (rango)
    """
    for col in ["sma20", "sma40", "sma200"]:
        if col in df.columns:
            df[f"{col}_slope"] = (df[col] - df[col].shift(1)).round(4)
    return df


def _calcular_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average True Range: mide la volatilidad media del mercado.
    True Range = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f"atr{period}"] = tr.rolling(window=period, min_periods=period).mean().round(4)
    df["atr14"] = df[f"atr{period}"]  # alias estándar
    return df


def _detectar_swings(df: pd.DataFrame, n: int = SWING_VELAS) -> pd.DataFrame:
    """
    Detecta Máximos y Mínimos Relevantes según la definición
    exacta del Método Power 4:

    MÁXIMO RELEVANTE: High[i] > High[i±1], High[i±2], High[i±3]
    MÍNIMO RELEVANTE: Low[i]  < Low[i±1],  Low[i±2],  Low[i±3]

    IMPORTANTE: Solo se puede confirmar un swing tras el cierre
    de las N velas posteriores → los últimos N valores son NaN.
    """
    highs = df["high"].values
    lows  = df["low"].values
    size  = len(df)

    swing_high       = np.zeros(size, dtype=bool)
    swing_low        = np.zeros(size, dtype=bool)
    swing_high_price = np.full(size, np.nan)
    swing_low_price  = np.full(size, np.nan)

    # Rango válido: necesitamos N velas antes Y N velas después
    for i in range(n, size - n):
        h = highs[i]
        l = lows[i]

        # Verificar N velas anteriores y N velas posteriores
        prev_highs = highs[i - n : i]
        post_highs = highs[i + 1 : i + n + 1]
        prev_lows  = lows[i - n : i]
        post_lows  = lows[i + 1 : i + n + 1]

        # Máximo relevante: estrictamente mayor que todos sus vecinos
        if h > prev_highs.max() and h > post_highs.max():
            swing_high[i]       = True
            swing_high_price[i] = round(float(h), 4)

        # Mínimo relevante: estrictamente menor que todos sus vecinos
        if l < prev_lows.min() and l < post_lows.min():
            swing_low[i]       = True
            swing_low_price[i] = round(float(l), 4)

    df["swing_high"]       = swing_high
    df["swing_low"]        = swing_low
    df["swing_high_price"] = swing_high_price
    df["swing_low_price"]  = swing_low_price

    return df


def _clasificar_velas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica cada vela según su estructura.
    body_pct = tamaño del cuerpo / rango total (0-1)
    doji cuando body_pct < 10%
    """
    rango = df["high"] - df["low"]
    cuerpo = (df["close"] - df["open"]).abs()

    df["vela_alcista"] = df["close"] > df["open"]
    df["vela_bajista"] = df["close"] < df["open"]
    df["body_pct"]     = (cuerpo / rango.replace(0, np.nan)).round(3)
    df["vela_doji"]    = df["body_pct"] < 0.1

    return df


# ══════════════════════════════════════════════════════════════════
#  FUNCIONES DE CONSULTA (usadas por otros módulos)
# ══════════════════════════════════════════════════════════════════

def get_ultimos_swing_highs(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """
    Devuelve los últimos N máximos relevantes confirmados.
    Útil para: verificar si los máximos son decrecientes (Etapa 4).
    """
    mask = df["swing_high"] == True
    return df.loc[mask, "swing_high_price"].dropna().tail(n)


def get_ultimos_swing_lows(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """
    Devuelve los últimos N mínimos relevantes confirmados.
    Útil para: verificar si los mínimos son crecientes (Etapa 2).
    """
    mask = df["swing_low"] == True
    return df.loc[mask, "swing_low_price"].dropna().tail(n)


def maximos_son_decrecientes(df: pd.DataFrame, n: int = 3) -> bool:
    """
    True si los últimos N máximos relevantes son estrictamente
    decrecientes. Condición necesaria para Etapa 4.
    """
    swings = get_ultimos_swing_highs(df, n)
    if len(swings) < n:
        return False
    vals = swings.values
    return all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def minimos_son_crecientes(df: pd.DataFrame, n: int = 3) -> bool:
    """
    True si los últimos N mínimos relevantes son estrictamente
    crecientes. Condición necesaria para Etapa 2.
    """
    swings = get_ultimos_swing_lows(df, n)
    if len(swings) < n:
        return False
    vals = swings.values
    return all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


def contar_cruces_sma20(df: pd.DataFrame, lookback: int = 20) -> int:
    """
    Cuenta cuántas veces el precio ha cruzado la SMA20
    en las últimas `lookback` velas.
    Usado para detectar rangos laterales (Etapas 1 y 3).
    """
    ventana = df.tail(lookback)
    if len(ventana) < 2 or "sma20" not in ventana.columns:
        return 0

    sobre_sma = ventana["close"] > ventana["sma20"]
    # Un cruce ocurre cuando cambia el lado respecto a la vela anterior
    cruces = (sobre_sma != sobre_sma.shift(1)).sum()
    return int(cruces)


def pct_cierres_sobre_sma20(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Porcentaje de velas cuyo Close está sobre la SMA20
    en la ventana de `lookback` velas.
    Etapa 2 requiere ≥ 90-95%.
    """
    ventana = df.tail(lookback)
    if len(ventana) == 0 or "sma20" not in ventana.columns:
        return 0.0
    sobre = (ventana["close"] > ventana["sma20"]).sum()
    return round(sobre / len(ventana), 3)


def pct_cierres_bajo_sma20(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Porcentaje de velas cuyo Close está bajo la SMA20.
    Etapa 4 requiere ≥ 90-95%.
    """
    return round(1.0 - pct_cierres_sobre_sma20(df, lookback), 3)


def resumen_indicadores(df: pd.DataFrame) -> dict:
    """
    Devuelve un dict con el estado actual de todos los
    indicadores en la última vela. Útil para logging y dashboard.
    """
    if df is None or len(df) == 0:
        return {}

    ult = df.iloc[-1]
    return {
        "close":          round(float(ult["close"]), 4),
        "sma20":          round(float(ult.get("sma20", 0)), 4),
        "sma40":          round(float(ult.get("sma40", 0)), 4),
        "sma200":         round(float(ult.get("sma200", 0)), 4),
        "sma20_slope":    round(float(ult.get("sma20_slope", 0)), 4),
        "sma40_slope":    round(float(ult.get("sma40_slope", 0)), 4),
        "atr14":          round(float(ult.get("atr14", 0)), 4),
        "dist_sma20_pct": round(float(ult.get("dist_sma20_pct", 0)), 2),
        "cerca_sma20":    bool(ult.get("cerca_sma20", False)),
        "vela_alcista":   bool(ult.get("vela_alcista", False)),
        "swing_highs_total": int(df["swing_high"].sum()),
        "swing_lows_total":  int(df["swing_low"].sum()),
        "cruces_sma20_20v":  contar_cruces_sma20(df, 20),
        "pct_sobre_sma20":   pct_cierres_sobre_sma20(df, 20),
        "mins_crecientes":   minimos_son_crecientes(df),
        "maxs_decrecientes": maximos_son_decrecientes(df),
    }
