"""
power4_bot/engine/stage_classifier.py
================================================
Clasificador de las 4 Etapas del Método Power 4.

Para cada activo y timeframe determina en qué
etapa se encuentra el mercado:

  Etapa 1 — Acumulación (Base / Rango lateral)
  Etapa 2 — Tendencia Alcista
  Etapa 3 — Distribución (Techo / Rango lateral)
  Etapa 4 — Tendencia Bajista

Y verifica el ALINEAMIENTO entre timeframe
semanal (W1) y diario (D1) para habilitar
señales de trading.
================================================
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import pandas as pd

from engine.indicators import (
    calcular_indicadores,
    contar_cruces_sma20,
    maximos_son_decrecientes,
    minimos_son_crecientes,
    pct_cierres_bajo_sma20,
    pct_cierres_sobre_sma20,
    resumen_indicadores,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  TIPOS Y CONSTANTES
# ══════════════════════════════════════════════════════════════════

class Etapa(IntEnum):
    DESCONOCIDA  = 0
    ACUMULACION  = 1
    ALCISTA      = 2
    DISTRIBUCION = 3
    BAJISTA      = 4


class Direccion:
    LONG  = "LONG"
    SHORT = "SHORT"
    NINGUNA = None


# Umbrales — alineados con settings.yaml
LOOKBACK              = 20     # velas para evaluar etapa
MIN_PCT_SOBRE_SMA20   = 0.90   # ≥90% cierres sobre SMA20 → Etapa 2
MIN_PCT_BAJO_SMA20    = 0.90   # ≥90% cierres bajo SMA20  → Etapa 4
MIN_CRUCES_RANGO      = 3      # ≥3 cruces SMA20 → rango lateral
UMBRAL_SLOPE_POSITIVO = 0.0005 # pendiente positiva
UMBRAL_SLOPE_NEGATIVO = -0.0005
UMBRAL_SLOPE_PLANO    = 0.0003


@dataclass
class ResultadoEtapa:
    """Resultado completo del análisis de etapa para un activo/timeframe."""
    etapa:            Etapa   = Etapa.DESCONOCIDA
    sma20_slope:      float   = 0.0
    sma40_slope:      float   = 0.0
    pct_sobre_sma20:  float   = 0.0
    cruces_sma20:     int     = 0
    mins_crecientes:  bool    = False
    maxs_decrecientes:bool    = False
    ultimo_close:     float   = 0.0
    ultimo_sma20:     float   = 0.0
    dist_sma20_pct:   float   = 0.0
    cerca_sma20:      bool    = False
    invalidada:       bool    = False   # True si se perforó el último mínimo relevante
    razon:            str     = ""      # Explicación textual de la clasificación


@dataclass
class ResultadoAlineamiento:
    """Resultado del alineamiento W1 + D1 para un símbolo."""
    symbol:       str
    etapa_w1:     Etapa           = Etapa.DESCONOCIDA
    etapa_d1:     Etapa           = Etapa.DESCONOCIDA
    direccion:    Optional[str]   = None   # "LONG" | "SHORT" | None
    alineado:     bool            = False
    cerca_sma20:  bool            = False  # del timeframe diario
    dist_sma20:   float           = 0.0
    operable:     bool            = False  # alineado AND cerca_sma20
    detalle_w1:   Optional[ResultadoEtapa] = None
    detalle_d1:   Optional[ResultadoEtapa] = None


# ══════════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL: CLASIFICAR ETAPA
# ══════════════════════════════════════════════════════════════════

def clasificar_etapa(
    df: pd.DataFrame,
    lookback: int = LOOKBACK,
) -> ResultadoEtapa:
    """
    Analiza las últimas `lookback` velas de un DataFrame
    ya enriquecido con indicadores y determina la etapa actual.

    El DataFrame DEBE haber pasado por calcular_indicadores().
    Si no tiene las columnas necesarias, las calcula internamente.

    Args:
        df:       DataFrame OHLC enriquecido con indicadores
        lookback: Número de velas para evaluar

    Returns:
        ResultadoEtapa con la etapa y todos los datos de soporte
    """
    resultado = ResultadoEtapa()

    # ── Guardar-calcular indicadores si faltan ────────────────────
    if "sma20" not in df.columns:
        df = calcular_indicadores(df)

    # Descartar NaN al inicio (los primeros ~200 no tienen SMA200)
    cols_presentes = [c for c in ["sma20", "sma40"] if c in df.columns]
    if not cols_presentes:
        resultado.razon = "Indicadores no calculados y datos insuficientes"
        return resultado
    df_valido = df.dropna(subset=cols_presentes)
    if len(df_valido) < lookback:
        resultado.razon = f"Datos insuficientes ({len(df_valido)} filas válidas)"
        return resultado

    # ── Métricas de la ventana de lookback ────────────────────────
    ventana = df_valido.tail(lookback)
    ultimo  = df_valido.iloc[-1]

    sma20_slope = float(ultimo.get("sma20_slope", 0) or 0)
    sma40_slope = float(ultimo.get("sma40_slope", 0) or 0)

    pct_sobre   = pct_cierres_sobre_sma20(df_valido, lookback)
    pct_bajo    = pct_cierres_bajo_sma20(df_valido, lookback)
    cruces      = contar_cruces_sma20(df_valido, lookback)
    mins_crec   = minimos_son_crecientes(df_valido, n=3)
    maxs_decr   = maximos_son_decrecientes(df_valido, n=3)
    close       = float(ultimo["close"])
    sma20       = float(ultimo["sma20"])
    dist_pct    = float(ultimo.get("dist_sma20_pct", abs(close - sma20) / sma20 * 100))
    cerca       = bool(ultimo.get("cerca_sma20", dist_pct < 4.0))

    # Rellenar campos comunes del resultado
    resultado.sma20_slope      = sma20_slope
    resultado.sma40_slope      = sma40_slope
    resultado.pct_sobre_sma20  = pct_sobre
    resultado.cruces_sma20     = cruces
    resultado.mins_crecientes  = mins_crec
    resultado.maxs_decrecientes= maxs_decr
    resultado.ultimo_close     = close
    resultado.ultimo_sma20     = sma20
    resultado.dist_sma20_pct   = dist_pct
    resultado.cerca_sma20      = cerca

    # ── ETAPA 2: Tendencia Alcista ────────────────────────────────
    #   • SMA20 y SMA40 apuntan arriba
    #   • ≥90% cierres sobre SMA20
    #   • Mínimos relevantes crecientes
    #   • INVALIDACIÓN: último cierre < último mínimo relevante
    if (
        sma20_slope > UMBRAL_SLOPE_POSITIVO
        and sma40_slope > UMBRAL_SLOPE_NEGATIVO   # SMA40 al menos no muy bajista
        and pct_sobre >= MIN_PCT_SOBRE_SMA20
        and mins_crec
    ):
        # Verificar invalidación de Etapa 2
        invalidada, razon_inv = _verificar_invalidacion_e2(df_valido)
        if invalidada:
            resultado.invalidada = True
            resultado.razon = f"Etapa 2 invalidada: {razon_inv}"
            # No asignamos Etapa 2, caemos al siguiente bloque
        else:
            resultado.etapa = Etapa.ALCISTA
            resultado.razon = (
                f"E2: slope20={sma20_slope:+.4f} slope40={sma40_slope:+.4f} "
                f"sobre_sma20={pct_sobre:.0%} mins_crec=True"
            )
            return resultado

    # ── ETAPA 4: Tendencia Bajista ────────────────────────────────
    #   • SMA20 y SMA40 apuntan abajo
    #   • ≥90% cierres bajo SMA20
    #   • Máximos relevantes decrecientes
    if (
        sma20_slope < UMBRAL_SLOPE_NEGATIVO
        and sma40_slope < UMBRAL_SLOPE_POSITIVO   # SMA40 al menos no muy alcista
        and pct_bajo >= MIN_PCT_BAJO_SMA20
        and maxs_decr
    ):
        # Verificar invalidación de Etapa 4
        invalidada, razon_inv = _verificar_invalidacion_e4(df_valido)
        if invalidada:
            resultado.invalidada = True
            resultado.razon = f"Etapa 4 invalidada: {razon_inv}"
        else:
            resultado.etapa = Etapa.BAJISTA
            resultado.razon = (
                f"E4: slope20={sma20_slope:+.4f} slope40={sma40_slope:+.4f} "
                f"bajo_sma20={pct_bajo:.0%} maxs_decr=True"
            )
            return resultado

    # ── ETAPA 1 o 3: Rango lateral ───────────────────────────────
    #   • SMA20 plana O precio cruza SMA20 repetidamente (≥3 cruces)
    #   • Condición relajada: si hay ≥3 cruces es rango independientemente
    #     del slope (el slope sinusoidal puede dar valores altos en la media)
    es_rango = (
        (abs(sma20_slope) < UMBRAL_SLOPE_PLANO and cruces >= MIN_CRUCES_RANGO)
        or (cruces >= MIN_CRUCES_RANGO and pct_sobre < 0.75 and pct_sobre > 0.25)
    )
    if es_rango:
        etapa_rango = _discriminar_e1_e3(df_valido, ventana)
        resultado.etapa = etapa_rango
        resultado.razon = (
            f"{'E1' if etapa_rango == Etapa.ACUMULACION else 'E3'}: "
            f"slope20={sma20_slope:+.4f} cruces={cruces} "
            f"(plana+rango lateral)"
        )
        return resultado

    # ── Fallback: asignar la etapa más probable ──────────────────
    resultado.etapa = _etapa_por_contexto(
        sma20_slope, sma40_slope, pct_sobre, pct_bajo, cruces
    )
    resultado.razon = (
        f"Fallback: slope20={sma20_slope:+.4f} "
        f"sobre={pct_sobre:.0%} bajo={pct_bajo:.0%} cruces={cruces}"
    )
    return resultado


# ══════════════════════════════════════════════════════════════════
#  VERIFICACIÓN DE ALINEAMIENTO (LA REGLA DE ORO)
# ══════════════════════════════════════════════════════════════════

def verificar_alineamiento(
    symbol:  str,
    df_w1:   pd.DataFrame,
    df_d1:   pd.DataFrame,
    lookback: int = LOOKBACK,
) -> ResultadoAlineamiento:
    """
    Aplica la Regla de Oro del Método Power 4:

      LONG  = Etapa Semanal 2 + Etapa Diaria 2
      SHORT = Etapa Semanal 4 + Etapa Diaria 4

    Solo si hay alineamiento Y el precio está a <4% de la SMA20
    diaria (Regla Barrio Sésamo) se marca como OPERABLE.

    Args:
        symbol:   Nombre del activo (para logging)
        df_w1:    DataFrame semanal con indicadores
        df_d1:    DataFrame diario con indicadores
        lookback: Ventana de evaluación

    Returns:
        ResultadoAlineamiento completo
    """
    res = ResultadoAlineamiento(symbol=symbol)

    # Calcular etapa en cada timeframe
    res.detalle_w1 = clasificar_etapa(df_w1, lookback)
    res.detalle_d1 = clasificar_etapa(df_d1, lookback)
    res.etapa_w1   = res.detalle_w1.etapa
    res.etapa_d1   = res.detalle_d1.etapa

    # Datos de distancia a SMA20 (del diario, que es el trigger)
    res.cerca_sma20 = res.detalle_d1.cerca_sma20
    res.dist_sma20  = res.detalle_d1.dist_sma20_pct

    # ── Evaluar alineamiento ─────────────────────────────────────
    # FASE TENDENCIAL (2 y 4)
    if res.etapa_w1 == Etapa.ALCISTA and res.etapa_d1 == Etapa.ALCISTA:
        res.alineado  = True
        res.direccion = Direccion.LONG
    elif res.etapa_w1 == Etapa.BAJISTA and res.etapa_d1 == Etapa.BAJISTA:
        res.alineado  = True
        res.direccion = Direccion.SHORT

    # FASE DE CONSOLIDACIÓN (1 y 3) - Nueva Regla para Acunamiento
    elif res.etapa_w1 == Etapa.ACUMULACION and res.etapa_d1 == Etapa.ACUMULACION:
        res.alineado  = True
        res.direccion = Direccion.LONG   # Acunamiento alcista en acumulación
    elif res.etapa_w1 == Etapa.DISTRIBUCION and res.etapa_d1 == Etapa.DISTRIBUCION:
        res.alineado  = True
        res.direccion = Direccion.SHORT  # Acunamiento bajista en distribución

    # ── OPERABLE: alineado + cerca de SMA20 ─────────────────────
    # Nota: Acunamiento TIENE su propio filtro de cercanía interno, 
    # pero aquí mantenemos la coherencia del sistema.
    res.operable = res.alineado and (res.cerca_sma20 or "ACU" in res.etapa_d1.name)

    logger.info(
        f"{symbol:>6} | "
        f"W1:{res.etapa_w1.value}({res.etapa_w1.name[:3]}) "
        f"D1:{res.etapa_d1.value}({res.etapa_d1.name[:3]}) | "
        f"{'✅ ' + (res.direccion or '') if res.alineado else '❌ ---':12} | "
        f"SMA20 dist:{res.dist_sma20:.1f}% "
        f"{'✓ OPERABLE' if res.operable else ''}"
    )

    return res


def analizar_watchlist(
    datos: dict,
    lookback: int = LOOKBACK,
) -> list:
    """
    Analiza todos los activos de la watchlist.

    Args:
        datos: dict con estructura:
               {symbol: {"W1": df_semanal, "D1": df_diario}}
        lookback: ventana de evaluación

    Returns:
        Lista de ResultadoAlineamiento ordenada:
        primero OPERABLE, luego ALINEADO, luego el resto.
    """
    resultados = []

    for symbol, tfs in datos.items():
        df_w1 = tfs.get("W1")
        df_d1 = tfs.get("D1")

        if df_w1 is None or df_d1 is None:
            logger.warning(f"{symbol}: faltan datos W1 o D1, saltando.")
            continue

        # Calcular indicadores si no están calculados
        if "sma20" not in df_w1.columns:
            df_w1 = calcular_indicadores(df_w1)
        if "sma20" not in df_d1.columns:
            df_d1 = calcular_indicadores(df_d1)

        res = verificar_alineamiento(symbol, df_w1, df_d1, lookback)
        resultados.append(res)

    # Ordenar: operables primero, luego alineados, luego el resto
    resultados.sort(
        key=lambda r: (
            not r.operable,
            not r.alineado,
            r.dist_sma20,
        )
    )

    # Log resumen
    operables = sum(1 for r in resultados if r.operable)
    alineados = sum(1 for r in resultados if r.alineado and not r.operable)
    logger.info(
        f"Watchlist analizada: {len(resultados)} símbolos | "
        f"Operables: {operables} | "
        f"Alineados (lejos SMA): {alineados}"
    )

    return resultados


# ══════════════════════════════════════════════════════════════════
#  HELPERS INTERNOS
# ══════════════════════════════════════════════════════════════════

def _verificar_invalidacion_e2(df: pd.DataFrame) -> tuple[bool, str]:
    """
    La Etapa 2 se invalida en el instante en que un cierre
    perfora a la baja el último mínimo relevante confirmado.

    Returns: (invalidada: bool, razon: str)
    """
    from engine.indicators import get_ultimos_swing_lows
    swings = get_ultimos_swing_lows(df, n=1)

    if len(swings) == 0:
        return False, ""

    ultimo_swing_low = swings.iloc[-1]
    ultimo_close     = float(df["close"].iloc[-1])

    if ultimo_close < ultimo_swing_low:
        return True, (
            f"Close {ultimo_close:.4f} < "
            f"último swing low {ultimo_swing_low:.4f}"
        )

    return False, ""


def _verificar_invalidacion_e4(df: pd.DataFrame) -> tuple[bool, str]:
    """
    La Etapa 4 se invalida si un cierre perfora al alza
    el último máximo relevante confirmado.
    """
    from engine.indicators import get_ultimos_swing_highs
    swings = get_ultimos_swing_highs(df, n=1)

    if len(swings) == 0:
        return False, ""

    ultimo_swing_high = swings.iloc[-1]
    ultimo_close      = float(df["close"].iloc[-1])

    if ultimo_close > ultimo_swing_high:
        return True, (
            f"Close {ultimo_close:.4f} > "
            f"último swing high {ultimo_swing_high:.4f}"
        )

    return False, ""


def _discriminar_e1_e3(
    df_completo: pd.DataFrame,
    ventana: pd.DataFrame,
) -> Etapa:
    """
    Diferencia Etapa 1 (acumulación) de Etapa 3 (distribución).

    Heurística:
    - Si el precio está en zona relativamente alta respecto a la
      SMA200 → probablemente E3 (viene de tendencia alcista)
    - Si el precio está en zona baja o SMA200 no tiene pendiente
      → probablemente E1
    - Si la SMA40 sigue siendo alcista leve → E3
    - Si la SMA40 es plana o bajista → E1
    """
    ultimo = df_completo.iloc[-1]

    sma40_slope  = float(ultimo.get("sma40_slope", 0) or 0)
    close        = float(ultimo["close"])
    sma200       = float(ultimo.get("sma200", 0) or 0)

    # Si precio > SMA200 y SMA40 aún con slope positivo leve → E3
    if sma200 > 0 and close > sma200 and sma40_slope > -UMBRAL_SLOPE_POSITIVO:
        return Etapa.DISTRIBUCION

    return Etapa.ACUMULACION


def _etapa_por_contexto(
    sma20_slope: float,
    sma40_slope: float,
    pct_sobre: float,
    pct_bajo: float,
    cruces: int,
) -> Etapa:
    """
    Fallback cuando ninguna condición principal se cumple.
    Asigna la etapa más probable por contexto parcial.
    """
    # Inclinación positiva dominante → tendencia alcista débil
    if sma20_slope > 0 and pct_sobre > 0.60:
        return Etapa.ALCISTA

    # Inclinación negativa dominante → tendencia bajista débil
    if sma20_slope < 0 and pct_bajo > 0.60:
        return Etapa.BAJISTA

    # Muchos cruces → rango lateral
    if cruces >= 2:
        return Etapa.ACUMULACION

    return Etapa.DESCONOCIDA


# ══════════════════════════════════════════════════════════════════
#  TABLA RESUMEN (para consola y dashboard)
# ══════════════════════════════════════════════════════════════════

def imprimir_tabla_resumen(resultados: list) -> None:
    """Imprime la tabla de etapas en consola."""
    cabecera = (
        f"{'SÍMBOLO':>8} │ {'E.SEM':>5} │ {'E.DIA':>5} │ "
        f"{'ALIN':>8} │ {'DIST SMA20':>10} │ {'OPERABLE':>8}"
    )
    separador = "─" * len(cabecera)

    print()
    print(separador)
    print(cabecera)
    print(separador)

    for r in resultados:
        alin_str = (
            f"✅ {r.direccion}" if r.alineado
            else "❌  ---"
        )
        operable_str = "⚡ SÍ" if r.operable else "   -"
        print(
            f"{r.symbol:>8} │ "
            f"E{r.etapa_w1.value:>4} │ "
            f"E{r.etapa_d1.value:>4} │ "
            f"{alin_str:>8} │ "
            f"{r.dist_sma20:>9.1f}% │ "
            f"{operable_str:>8}"
        )

    print(separador)
    operables = sum(1 for r in resultados if r.operable)
    alineados = sum(1 for r in resultados if r.alineado)
    print(
        f"  Total: {len(resultados)} │ "
        f"Alineados: {alineados} │ "
        f"Operables: {operables}"
    )
    print(separador)
    print()
