"""
power4_bot/engine/patterns/patron_123.py
================================================
Patrón 1-2-3 Alcista y Bajista

ALCISTA (Etapa 2):
  Vela 1: Gran vela verde sólida tras baja volatilidad
          body > percentil 80 de los últimos 20 cuerpos
  Vela 2: Inside bar o vela pequeña que retrocede
          ≤ 30% del cuerpo de Vela 1 (tercio superior)
          Máximos de V1 y V2 relativamente alineados
  Trigger: Precio supera max(High_V1, High_V2)

BAJISTA (Etapa 4): imagen simétrica con vela roja
================================================
"""

import logging
from typing import Optional, Union, TYPE_CHECKING
import pandas as pd
import numpy as np

from engine.patterns.base import PatronBase, Señal, SEÑAL_VACIA

logger = logging.getLogger(__name__)

MAX_RETROCESO_V2   = 0.30   # Vela 2 no puede retroceder más del 30% del cuerpo V1
MIN_BODY_PERCENTIL = 80     # Vela 1 debe estar en el percentil 80 de cuerpos recientes
LOOKBACK_PERCENTIL = 20     # Velas para calcular el percentil del cuerpo


class Patron123Alcista(PatronBase):
    """Patrón 1-2-3 Alcista — LONG en Etapa 2."""

    nombre    = "123_ALC"
    direccion = "LONG"

    def detectar(self, df: pd.DataFrame) -> Señal:
        if len(df) < LOOKBACK_PERCENTIL + 3:
            return SEÑAL_VACIA

        n   = len(df)
        idx = n - 1

        # Las tres velas del patrón son las últimas 3
        # Vela actual = posible trigger o vela 2/3
        # Buscamos V1 en las últimas 5 velas
        for offset in range(1, 5):
            idx_v1 = idx - offset
            idx_v2 = idx_v1 + 1
            idx_trigger = idx

            if idx_v1 < LOOKBACK_PERCENTIL:
                break

            señal = self._evaluar_123(df, idx_v1, idx_v2, idx_trigger)
            if señal.detectado:
                return señal

        return SEÑAL_VACIA

    def detectar_prepatron(self, df: pd.DataFrame) -> "Señal | None":
        """
        Pre-patrón 123 Alcista: V1 y V2 formadas, trigger pendiente.
        """
        if len(df) < LOOKBACK_PERCENTIL + 2:
            return None

        n = len(df)
        idx_v2 = n - 1
        idx_v1 = idx_v2 - 1

        # Evaluar si V1 y V2 cumplen las condiciones, usando idx_v2 como "trigger virtual"
        # para verificar si el precio actual de V2 ya superó el nivel (en cuyo caso no es PRE)
        v1 = df.iloc[idx_v1]
        v2 = df.iloc[idx_v2]

        body_v1 = float(v1["close"]) - float(v1["open"])
        if body_v1 <= 0: return None

        cuerpos_previos = (
            df["close"].iloc[idx_v1 - LOOKBACK_PERCENTIL : idx_v1]
            - df["open"].iloc[idx_v1 - LOOKBACK_PERCENTIL : idx_v1]
        ).abs()
        if len(cuerpos_previos) == 0: return None

        percentil_80 = np.percentile(cuerpos_previos, MIN_BODY_PERCENTIL)
        if body_v1 < percentil_80: return None

        high_v1 = float(v1["high"])
        low_v2  = float(v2["low"])
        high_v2 = float(v2["high"])

        retroceso = (high_v1 - low_v2) / body_v1
        if retroceso > MAX_RETROCESO_V2: return None
        if high_v2 > high_v1 * 1.002: return None

        nivel_trigger = max(high_v1, high_v2)
        precio_actual = float(v2["close"])

        # Si ya cerró por encima, no es PRE (lo detecta detectar normal)
        if precio_actual >= nivel_trigger:
            return None

        nivel_activacion = round(nivel_trigger * (1 + self.BUFFER_ENTRADA_PCT), 4)

        return Señal(
            detectado        = True,
            precio_entrada   = nivel_activacion,
            nivel_activacion = nivel_activacion,
            modo_entrada     = "condicional",
            pendiente_expira = 2,
            idx_vela         = idx_v2,
            razon            = (
                f"123_ALC PRE: V1 y V2 formadas. "
                f"BUY_STOP en {nivel_activacion:.4f} (valla de salida)."
            ),
            fase_mercado     = "Inhalación",
            datos_extra      = {"nivel_trigger": nivel_trigger}
        )

    def _evaluar_123(
        self,
        df:          pd.DataFrame,
        idx_v1:      int,
        idx_v2:      int,
        idx_trigger: int,
    ) -> Señal:
        v1 = df.iloc[idx_v1]
        v2 = df.iloc[idx_v2]
        vt = df.iloc[idx_trigger]

        # ── VELA 1: Gran vela verde sólida ────────────────────────────
        body_v1 = float(v1["close"]) - float(v1["open"])
        if body_v1 <= 0:
            return SEÑAL_VACIA   # Debe ser vela verde

        # Body en percentil 80 de los últimos N cuerpos
        cuerpos_previos = (
            df["close"].iloc[idx_v1 - LOOKBACK_PERCENTIL : idx_v1]
            - df["open"].iloc[idx_v1 - LOOKBACK_PERCENTIL : idx_v1]
        ).abs()

        if len(cuerpos_previos) == 0:
            return SEÑAL_VACIA

        percentil_80 = np.percentile(cuerpos_previos, MIN_BODY_PERCENTIL)
        if body_v1 < percentil_80:
            return SEÑAL_VACIA   # Vela 1 no es suficientemente grande

        # ── VELA 2: Inside bar o pequeña con retroceso ≤ 30% ──────────
        high_v1  = float(v1["high"])
        low_v1   = float(v1["low"])
        open_v1  = float(v1["open"])
        close_v1 = float(v1["close"])

        high_v2  = float(v2["high"])
        low_v2   = float(v2["low"])
        close_v2 = float(v2["close"])

        # Retroceso = cuánto baja V2 desde el máximo de V1
        retroceso = (high_v1 - low_v2) / body_v1 if body_v1 > 0 else 1.0

        if retroceso > MAX_RETROCESO_V2:
            return SEÑAL_VACIA   # Vela 2 retrocede demasiado

        # V2 debe estar contenida (no superar el máximo de V1)
        if high_v2 > high_v1 * 1.002:
            return SEÑAL_VACIA   # V2 supera V1, no es consolidación

        # ── TRIGGER: precio supera max(High_V1, High_V2) ──────────────
        nivel_trigger = max(high_v1, high_v2)
        precio_actual = float(vt["close"])

        if precio_actual <= nivel_trigger:
            return SEÑAL_VACIA

        precio_entrada = round(nivel_trigger * (1 + self.BUFFER_ENTRADA_PCT), 4)

        return Señal(
            detectado      = True,
            precio_entrada = precio_entrada,
            idx_vela       = idx_trigger,
            razon          = (
                f"123_ALC: V1 body={body_v1:.4f} (P{MIN_BODY_PERCENTIL}={percentil_80:.4f}), "
                f"retroceso V2={retroceso:.1%}. "
                f"Trigger: {nivel_trigger:.4f} superado."
            ),
            datos_extra = {
                "idx_v1":        idx_v1,
                "idx_v2":        idx_v2,
                "body_v1":       body_v1,
                "retroceso_v2":  retroceso,
                "nivel_trigger": nivel_trigger,
            }
        )


class Patron123Bajista(PatronBase):
    """Patrón 1-2-3 Bajista — SHORT en Etapa 4."""

    nombre    = "123_BAJ"
    direccion = "SHORT"

    def detectar(self, df: pd.DataFrame) -> Señal:
        if len(df) < LOOKBACK_PERCENTIL + 3:
            return SEÑAL_VACIA

        n   = len(df)
        idx = n - 1

        for offset in range(1, 5):
            idx_v1      = idx - offset
            idx_v2      = idx_v1 + 1
            idx_trigger = idx

            if idx_v1 < LOOKBACK_PERCENTIL:
                break

            señal = self._evaluar_123_baj(df, idx_v1, idx_v2, idx_trigger)
            if señal.detectado:
                return señal

        return SEÑAL_VACIA

    def detectar_prepatron(self, df: pd.DataFrame) -> "Señal | None":
        """
        Pre-patrón 123 Bajista: V1 y V2 formadas, trigger pendiente.
        """
        if len(df) < LOOKBACK_PERCENTIL + 2:
            return None

        n = len(df)
        idx_v2 = n - 1
        idx_v1 = idx_v2 - 1

        v1 = df.iloc[idx_v1]
        v2 = df.iloc[idx_v2]

        body_v1 = float(v1["open"]) - float(v1["close"])   # positivo si bajista
        if body_v1 <= 0: return None

        cuerpos_previos = (
            df["close"].iloc[idx_v1 - LOOKBACK_PERCENTIL : idx_v1]
            - df["open"].iloc[idx_v1 - LOOKBACK_PERCENTIL : idx_v1]
        ).abs()
        if len(cuerpos_previos) == 0: return None

        percentil_80 = np.percentile(cuerpos_previos, MIN_BODY_PERCENTIL)
        if body_v1 < percentil_80: return None

        low_v1  = float(v1["low"])
        high_v2 = float(v2["high"])
        low_v2  = float(v2["low"])

        # Retroceso: cuánto sube V2 desde el mínimo de V1
        retroceso = (high_v2 - low_v1) / body_v1 if body_v1 > 0 else 1.0
        if retroceso > MAX_RETROCESO_V2: return None
        if low_v2 < low_v1 * 0.998: return None

        # Nivel trigger: el mínimo de V1 o V2
        nivel_trigger = min(low_v1, low_v2)
        precio_actual = float(v2["close"])

        if precio_actual <= nivel_trigger:
            return None

        nivel_activacion = round(nivel_trigger * (1 - self.BUFFER_ENTRADA_PCT), 4)

        return Señal(
            detectado        = True,
            precio_entrada   = nivel_activacion,
            nivel_activacion = nivel_activacion,
            modo_entrada     = "condicional",
            pendiente_expira = 2,
            idx_vela         = idx_v2,
            razon            = (
                f"123_BAJ PRE: V1 y V2 formadas. "
                f"SELL_STOP en {nivel_activacion:.4f} (valla de salida)."
            ),
            fase_mercado     = "Inhalación",
            datos_extra      = {"nivel_trigger": nivel_trigger}
        )

    def _evaluar_123_baj(
        self,
        df:          pd.DataFrame,
        idx_v1:      int,
        idx_v2:      int,
        idx_trigger: int,
    ) -> Señal:

        v1 = df.iloc[idx_v1]
        v2 = df.iloc[idx_v2]
        vt = df.iloc[idx_trigger]

        # ── VELA 1: Gran vela roja sólida ─────────────────────────────
        body_v1 = float(v1["open"]) - float(v1["close"])   # positivo si bajista
        if body_v1 <= 0:
            return SEÑAL_VACIA

        cuerpos_previos = (
            df["close"].iloc[idx_v1 - LOOKBACK_PERCENTIL : idx_v1]
            - df["open"].iloc[idx_v1 - LOOKBACK_PERCENTIL : idx_v1]
        ).abs()

        percentil_80 = np.percentile(cuerpos_previos, MIN_BODY_PERCENTIL)
        if body_v1 < percentil_80:
            return SEÑAL_VACIA

        # ── VELA 2: Contenida en tercio superior de V1 ────────────────
        low_v1   = float(v1["low"])
        high_v1  = float(v1["high"])
        low_v2   = float(v2["low"])
        high_v2  = float(v2["high"])

        # Retroceso: cuánto sube V2 desde el mínimo de V1
        retroceso = (high_v2 - low_v1) / body_v1 if body_v1 > 0 else 1.0

        if retroceso > MAX_RETROCESO_V2:
            return SEÑAL_VACIA

        if low_v2 < low_v1 * 0.998:
            return SEÑAL_VACIA   # V2 supera el mínimo de V1

        # ── TRIGGER: precio pierde min(Low_V1, Low_V2) ────────────────
        nivel_trigger = min(low_v1, low_v2)
        precio_actual = float(vt["close"])

        if precio_actual >= nivel_trigger:
            return SEÑAL_VACIA

        precio_entrada = round(nivel_trigger * (1 - self.BUFFER_ENTRADA_PCT), 4)

        return Señal(
            detectado      = True,
            precio_entrada = precio_entrada,
            idx_vela       = idx_trigger,
            razon          = (
                f"123_BAJ: V1 body={body_v1:.4f} (P{MIN_BODY_PERCENTIL}={percentil_80:.4f}), "
                f"retroceso V2={retroceso:.1%}. "
                f"Trigger: {nivel_trigger:.4f} perforado."
            ),
            datos_extra = {
                "idx_v1":        idx_v1,
                "idx_v2":        idx_v2,
                "body_v1":       body_v1,
                "retroceso_v2":  retroceso,
                "nivel_trigger": nivel_trigger,
            }
        )
