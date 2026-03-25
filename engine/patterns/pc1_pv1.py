"""
power4_bot/engine/patterns/pc1_pv1.py
================================================
PC1 — Patrón de Compra 1  (LONG, Etapa 2)
PV1 — Patrón de Venta 1   (SHORT, Etapa 4)

Ambos son imágenes simétricas:

PC1: Tras un nuevo máximo (exhalación), el precio
retrocede formando ≥3 MÁXIMOS DECRECIENTES hacia
la SMA20 (inhalación). El trigger es la ruptura
al alza del último máximo decreciente.

PV1: Tras un nuevo mínimo (exhalación), el precio
sube formando ≥3 MÍNIMOS CRECIENTES hacia la SMA20
(inhalación). El trigger es la perforación a la
baja del último mínimo creciente.
================================================
"""

import logging
import pandas as pd
import numpy as np

from engine.patterns.base import PatronBase, Señal, SEÑAL_VACIA

logger = logging.getLogger(__name__)

# Parámetros
MIN_ESCALONES = 3    # Mínimo de escalones en la escalera
LOOKBACK_MAX  = 30   # Velas hacia atrás para buscar el inicio del patrón


class PC1(PatronBase):
    """Patrón de Compra 1 — LONG en Etapa 2."""

    nombre    = "PC1"
    direccion = "LONG"

    def detectar(self, df: pd.DataFrame) -> Señal:
        """
        Busca una escalera de máximos decrecientes reciente
        que termine cerca de la SMA20. El trigger es superar
        el último máximo (el más bajo de la escalera).
        """
        highs  = df["high"].values
        closes = df["close"].values
        sma20  = df["sma20"].values
        n      = len(df)
        idx    = n - 1   # vela actual (índice del trigger)

        # ── 1. Buscar el swing high reciente (inicio de la inhalación) ──
        ventana_inicio = max(0, idx - LOOKBACK_MAX)
        swing_highs_idx = [
            i for i in range(ventana_inicio, idx - 1)
            if df["swing_high"].iloc[i]
        ]

        if not swing_highs_idx:
            return SEÑAL_VACIA

        inicio = swing_highs_idx[-1]

        # ── 2. Verificar escalera de máximos decrecientes ──────────────
        escalones = _encontrar_escalera_decreciente(highs, inicio, idx)

        if len(escalones) < MIN_ESCALONES:
            return SEÑAL_VACIA

        # ── 3. Precio actual cerca del último máximo de la escalera ────
        ultimo_maximo = highs[escalones[-1]]
        precio_actual = closes[idx]

        if precio_actual <= ultimo_maximo:
            return SEÑAL_VACIA   # Aún no hay trigger

        # ── 4. El trigger se activa: precio superó el último máximo ────
        precio_entrada = round(float(ultimo_maximo) * (1 + self.BUFFER_ENTRADA_PCT), 4)

        señal = Señal(
            detectado      = True,
            precio_entrada = precio_entrada,
            idx_vela       = idx,
            razon          = (
                f"PC1: escalera de {len(escalones)} máximos decrecientes. "
                f"Último máximo: {ultimo_maximo:.4f} superado."
            ),
            fase_mercado   = "Inhalación",
            datos_extra = {
                "escalones":     escalones,
                "inicio_patron": inicio,
                "ultimo_maximo": ultimo_maximo,
                "n_escalones":   len(escalones),
            }
        )
        return señal

    def detectar_prepatron(self, df) -> "Señal | None":
        """
        Pre-patrón PC1: escalera formada pero el precio AÚN no superó
        el último máximo decreciente → colocar BUY_STOP en ese nivel.
        """
        highs  = df["high"].values
        closes = df["close"].values
        n      = len(df)
        idx    = n - 1

        ventana_inicio = max(0, idx - LOOKBACK_MAX)
        swing_highs_idx = [
            i for i in range(ventana_inicio, idx - 1)
            if df["swing_high"].iloc[i]
        ]
        if not swing_highs_idx:
            return None

        inicio    = swing_highs_idx[-1]
        escalones = _encontrar_escalera_decreciente(highs, inicio, idx)

        if len(escalones) < MIN_ESCALONES:
            return None

        ultimo_maximo = highs[escalones[-1]]
        precio_actual = closes[idx]

        # Pre-patrón: precio AÚN por debajo del último máximo
        if precio_actual >= ultimo_maximo:
            return None   # Ya confirmado (lo detecta detectar())

        nivel_activacion = round(float(ultimo_maximo) * (1 + self.BUFFER_ENTRADA_PCT), 4)

        return Señal(
            detectado        = True,
            precio_entrada   = nivel_activacion,   # Precio si se activa
            nivel_activacion = nivel_activacion,   # Nivel del BUY_STOP
            modo_entrada     = "condicional",
            pendiente_expira = 3,
            idx_vela         = idx,
            razon            = (
                f"PC1 PRE: {len(escalones)} escalones formados. "
                f"BUY_STOP en {nivel_activacion:.4f} (último máximo aún no superado)."
            ),
            fase_mercado   = "Inhalación",
            datos_extra = {
                "escalones":     escalones,
                "ultimo_maximo": ultimo_maximo,
                "n_escalones":   len(escalones),
            }
        )




class PV1(PatronBase):
    """Patrón de Venta 1 — SHORT en Etapa 4."""

    nombre    = "PV1"
    direccion = "SHORT"

    def detectar(self, df: pd.DataFrame) -> Señal:
        """
        Busca una escalera de mínimos crecientes reciente
        que termine cerca de la SMA20. El trigger es perder
        el último mínimo (el más alto de la escalera).
        """
        lows   = df["low"].values
        closes = df["close"].values
        sma20  = df["sma20"].values
        n      = len(df)
        idx    = n - 1

        # ── 1. Buscar el swing low reciente ────────────────────────────
        ventana_inicio = max(0, idx - LOOKBACK_MAX)
        swing_lows_idx = [
            i for i in range(ventana_inicio, idx - 1)
            if df["swing_low"].iloc[i]
        ]

        if not swing_lows_idx:
            return SEÑAL_VACIA

        inicio = swing_lows_idx[-1]

        # ── 2. Verificar escalera de mínimos crecientes ────────────────
        escalones = _encontrar_escalera_creciente(lows, inicio, idx)

        if len(escalones) < MIN_ESCALONES:
            return SEÑAL_VACIA

        # ── 3. Precio perdió el último mínimo (trigger de venta) ───────
        ultimo_minimo = lows[escalones[-1]]
        precio_actual = closes[idx]

        if precio_actual >= ultimo_minimo:
            return SEÑAL_VACIA   # Aún no hay trigger

        # ── 4. Trigger activo ───────────────────────────────────────────
        precio_entrada = round(float(ultimo_minimo) * (1 - self.BUFFER_ENTRADA_PCT), 4)

        señal = Señal(
            detectado      = True,
            precio_entrada = precio_entrada,
            idx_vela       = idx,
            razon          = (
                f"PV1: escalera de {len(escalones)} mínimos crecientes. "
                f"Último mínimo: {ultimo_minimo:.4f} perforado."
            ),
            fase_mercado   = "Inhalación",
            datos_extra = {
                "escalones":     escalones,
                "inicio_patron": inicio,
                "ultimo_minimo": ultimo_minimo,
                "n_escalones":   len(escalones),
            }
        )
        return señal

    def detectar_prepatron(self, df) -> "Señal | None":
        """
        Pre-patrón PV1: escalera de mínimos crecientes formada pero el
        precio AÚN no ha perforado el último mínimo → SELL_STOP allí.
        """
        lows   = df["low"].values
        closes = df["close"].values
        n      = len(df)
        idx    = n - 1

        ventana_inicio = max(0, idx - LOOKBACK_MAX)
        swing_lows_idx = [
            i for i in range(ventana_inicio, idx - 1)
            if df["swing_low"].iloc[i]
        ]
        if not swing_lows_idx:
            return None

        inicio    = swing_lows_idx[-1]
        escalones = _encontrar_escalera_creciente(lows, inicio, idx)

        if len(escalones) < MIN_ESCALONES:
            return None

        ultimo_minimo = lows[escalones[-1]]
        precio_actual = closes[idx]

        # Pre-patrón: precio AÚN por encima del último mínimo
        if precio_actual <= ultimo_minimo:
            return None   # Ya confirmado

        nivel_activacion = round(float(ultimo_minimo) * (1 - self.BUFFER_ENTRADA_PCT), 4)

        return Señal(
            detectado        = True,
            precio_entrada   = nivel_activacion,
            nivel_activacion = nivel_activacion,
            modo_entrada     = "condicional",
            pendiente_expira = 3,
            idx_vela         = idx,
            razon            = (
                f"PV1 PRE: {len(escalones)} escalones formados. "
                f"SELL_STOP en {nivel_activacion:.4f} (último mínimo aún no perforado)."
            ),
            fase_mercado   = "Inhalación",
            datos_extra = {
                "escalones":     escalones,
                "ultimo_minimo": ultimo_minimo,
                "n_escalones":   len(escalones),
            }
        )



# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _encontrar_escalera_decreciente(
    highs: np.ndarray,
    inicio: int,
    fin: int,
) -> list:
    """
    A partir del swing high en `inicio`, encuentra la secuencia
    de velas donde cada máximo es menor que el anterior
    (escalera descendente de máximos = inhalación en PC1).

    Returns: lista de índices que forman la escalera
    """
    escalones = []
    prev_high = highs[inicio]

    for i in range(inicio + 1, fin):
        if highs[i] < prev_high:
            escalones.append(i)
            prev_high = highs[i]
        else:
            # Si rompe la estructura, reiniciar desde aquí
            if highs[i] > highs[inicio]:
                # Nuevo máximo mayor que el inicio → no es inhalación válida
                escalones = []
            prev_high = highs[i]

    return escalones


def _encontrar_escalera_creciente(
    lows: np.ndarray,
    inicio: int,
    fin: int,
) -> list:
    """
    A partir del swing low en `inicio`, encuentra la secuencia
    de velas donde cada mínimo es mayor que el anterior
    (escalera ascendente de mínimos = inhalación en PV1).

    Returns: lista de índices que forman la escalera
    """
    escalones = []
    prev_low = lows[inicio]

    for i in range(inicio + 1, fin):
        if lows[i] > prev_low:
            escalones.append(i)
            prev_low = lows[i]
        else:
            if lows[i] < lows[inicio]:
                escalones = []
            prev_low = lows[i]

    return escalones
