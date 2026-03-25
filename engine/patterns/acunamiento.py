"""
power4_bot/engine/patterns/acunamiento.py
================================================
Acunamiento Alcista  — LONG, transición E1→E2
Acunamiento Bajista  — SHORT, transición E3→E4

El PATRÓN ESTRELLA del Método Power 4.
Captura el día exacto en que termina el rango
y nace una nueva tendencia fluida.

ALCISTA:
  • Precio ha cruzado SMA20 ≥3 veces (rango lateral)
  • En el retroceso: Low ≤ SMA20 PERO Close > SMA20
    (la SMA "acuna" al precio, no lo deja cerrar debajo)
  • SMA20 apunta hacia arriba (pendiente positiva)
  • Vela = verde (alcista)

BAJISTA: imagen simétrica
  • High ≥ SMA20 PERO Close < SMA20
  • SMA20 apunta hacia abajo
  • Vela = roja (bajista)

NOTA: Este patrón NO aplica la Regla Barrio Sésamo
(MAX_DIST_SMA20_PCT = 0) porque por definición el
precio está tocando/cruzando la SMA20.
================================================
"""

import logging
import pandas as pd

from engine.indicators import contar_cruces_sma20
from engine.patterns.base import PatronBase, Señal, SEÑAL_VACIA

logger = logging.getLogger(__name__)

MIN_CRUCES        = 3    # Cruces mínimos para confirmar rango previo
LOOKBACK_CRUCES   = 20   # Ventana para contar cruces
SLOPE_MIN_ALCISTA =  0.0001   # SMA20 debe apuntar arriba (aunque sea levemente)
SLOPE_MAX_BAJISTA = -0.0001   # SMA20 debe apuntar abajo


class AcunamientoAlcista(PatronBase):
    """
    Acunamiento Alcista — LONG.
    La SMA20 acuna al precio: low toca/penetra SMA20
    pero el cierre queda por encima. SMA20 apunta arriba.
    """

    nombre             = "ACUN_ALC"
    direccion          = "LONG"
    MAX_DIST_SMA20_PCT = 0.0   # Sin restricción Barrio Sésamo (precio toca SMA)

    def _precondiciones(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Override: no aplica límite de distancia a SMA20."""
        if df is None or len(df) < 30:
            return False, "DataFrame insuficiente"
        if "sma20" not in df.columns:
            return False, "Indicadores no calculados"
        return True, ""

    def detectar(self, df: pd.DataFrame) -> Señal:
        n   = len(df)
        idx = n - 1

        ultimo = df.iloc[idx]
        sma20  = float(ultimo["sma20"])

        if pd.isna(sma20):
            return SEÑAL_VACIA

        low   = float(ultimo["low"])
        close = float(ultimo["close"])
        high  = float(ultimo["high"])

        # ── 1. Rango lateral previo: ≥ MIN_CRUCES cruces SMA20 ────────
        cruces = contar_cruces_sma20(df, LOOKBACK_CRUCES)
        if cruces < MIN_CRUCES:
            return SEÑAL_VACIA

        # ── 2. Condición de acunamiento: Low ≤ SMA20 Y Close > SMA20 ──
        toca_sma  = low <= sma20
        cierra_sobre = close > sma20
        if not (toca_sma and cierra_sobre):
            return SEÑAL_VACIA

        # ── 3. SMA20 apuntando hacia arriba ────────────────────────────
        slope = float(ultimo.get("sma20_slope", 0) or 0)
        if slope < SLOPE_MIN_ALCISTA:
            return SEÑAL_VACIA

        # ── 4. Vela alcista (cierre > apertura) ───────────────────────
        if not bool(ultimo.get("vela_alcista", close > float(ultimo["open"]))):
            return SEÑAL_VACIA

        # ── Trigger: Buy Stop sobre el máximo de la vela ──────────────
        precio_entrada = round(high * (1 + self.BUFFER_ENTRADA_PCT), 4)
        stop_loss      = round(low  * (1 - self.STOP_BUFFER_PCT), 4)

        return Señal(
            detectado      = True,
            precio_entrada = precio_entrada,
            stop_loss      = stop_loss,
            idx_vela       = idx,
            razon          = (
                f"ACUN_ALC: low={low:.4f} ≤ sma20={sma20:.4f}, "
                f"close={close:.4f} > sma20. "
                f"slope={slope:+.4f}. cruces={cruces}."
            ),
            datos_extra = {
                "sma20":   sma20,
                "slope":   slope,
                "cruces":  cruces,
                "low_vela": low,
                "high_vela": high,
            }
        )


class AcunamientoBajista(PatronBase):
    """
    Acunamiento Bajista — SHORT.
    El precio choca con SMA20 desde abajo: high toca/penetra
    SMA20 pero el cierre queda por debajo. SMA20 apunta abajo.
    """

    nombre             = "ACUN_BAJ"
    direccion          = "SHORT"
    MAX_DIST_SMA20_PCT = 0.0

    def _precondiciones(self, df: pd.DataFrame) -> tuple[bool, str]:
        if df is None or len(df) < 30:
            return False, "DataFrame insuficiente"
        if "sma20" not in df.columns:
            return False, "Indicadores no calculados"
        return True, ""

    def detectar(self, df: pd.DataFrame) -> Señal:
        n   = len(df)
        idx = n - 1

        ultimo = df.iloc[idx]
        sma20  = float(ultimo["sma20"])

        if pd.isna(sma20):
            return SEÑAL_VACIA

        low   = float(ultimo["low"])
        close = float(ultimo["close"])
        high  = float(ultimo["high"])

        # ── 1. Rango lateral previo ────────────────────────────────────
        cruces = contar_cruces_sma20(df, LOOKBACK_CRUCES)
        if cruces < MIN_CRUCES:
            return SEÑAL_VACIA

        # ── 2. Condición de rechazo: High ≥ SMA20 Y Close < SMA20 ─────
        toca_sma    = high >= sma20
        cierra_bajo = close < sma20
        if not (toca_sma and cierra_bajo):
            return SEÑAL_VACIA

        # ── 3. SMA20 apuntando hacia abajo ─────────────────────────────
        slope = float(ultimo.get("sma20_slope", 0) or 0)
        if slope > SLOPE_MAX_BAJISTA:
            return SEÑAL_VACIA

        # ── 4. Vela bajista ────────────────────────────────────────────
        if not bool(ultimo.get("vela_bajista", close < float(ultimo["open"]))):
            return SEÑAL_VACIA

        # ── Trigger: Sell Stop bajo el mínimo de la vela ──────────────
        precio_entrada = round(low  * (1 - self.BUFFER_ENTRADA_PCT), 4)
        stop_loss      = round(high * (1 + self.STOP_BUFFER_PCT), 4)

        return Señal(
            detectado      = True,
            precio_entrada = precio_entrada,
            stop_loss      = stop_loss,
            idx_vela       = idx,
            razon          = (
                f"ACUN_BAJ: high={high:.4f} ≥ sma20={sma20:.4f}, "
                f"close={close:.4f} < sma20. "
                f"slope={slope:+.4f}. cruces={cruces}."
            ),
            datos_extra = {
                "sma20":    sma20,
                "slope":    slope,
                "cruces":   cruces,
                "low_vela": low,
                "high_vela": high,
            }
        )
