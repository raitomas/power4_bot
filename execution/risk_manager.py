"""
power4_bot/execution/risk_manager.py
================================================
Gestión monetaria del Método Power 4.

Responsabilidades:
  1. Calcular el Stop Loss inicial (regla del 1%)
  2. Calcular el tamaño de posición (riesgo fijo)
  3. Validar que la distancia al SL es 6-9% del precio
  4. Calcular el Take Profit (zona de fricción previa)
  5. Verificar límites globales de riesgo (kill switch)

Regla del 1%:
  LONG:  ref = min(low_hoy, low_ayer)  → SL = ref × 0.99
  SHORT: ref = max(high_hoy, high_ayer) → SL = ref × 1.01

Sizing:
  riesgo_$  = capital × riesgo_pct        (ej. 100k × 0.005 = $500)
  distancia = |precio_entrada - stop_loss|
  nº_acc    = floor(riesgo_$ / distancia)
================================================
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from engine.patterns.base import Señal

logger = logging.getLogger(__name__)

# ── Parámetros por defecto (sobreescribibles desde settings.yaml) ─
CAPITAL_DEFAULT        = 100_000.0
RIESGO_POR_OP          = 0.005      # 0.5% del capital
STOP_BUFFER_PCT        = 0.01       # 1% buffer sobre/bajo el ref
DIST_SL_MIN_PCT        = 0.04       # SL mínimo al 4% del precio
DIST_SL_MAX_PCT        = 0.10       # SL máximo al 10% del precio
MAX_POSICIONES         = 5
MAX_DRAWDOWN_KILL      = 0.08       # Kill switch al -8%
MIN_RATIO_RR           = 1.5        # Beneficio mínimo / Riesgo


@dataclass
class OrdenCalculada:
    """
    Resultado completo del cálculo de riesgo para una señal.
    Lista para ser enviada al order manager.
    """
    symbol:          str   = ""
    direccion:       str   = ""        # "LONG" | "SHORT"
    patron:          str   = ""
    precio_entrada:  float = 0.0       # Precio de la orden pendiente
    stop_loss:       float = 0.0       # Stop Loss calculado
    take_profit:     float = 0.0       # Take Profit calculado
    volumen:         float = 0.0       # Nº de acciones/lotes
    riesgo_dolares:  float = 0.0       # Riesgo máximo en $
    distancia_sl_pct:float = 0.0       # % de distancia entre entrada y SL
    ratio_rr:        float = 0.0       # R/R estimado
    valida:          bool  = False     # Pasa todos los filtros
    motivo_rechazo:  str   = ""        # Si valida=False, explica por qué

    def __repr__(self):
        estado = "✅" if self.valida else "❌"
        if not self.valida:
            return f"Orden({estado} {self.symbol} [{self.patron}] — {self.motivo_rechazo})"
        return (
            f"Orden({estado} {self.symbol} {self.direccion} [{self.patron}] | "
            f"entrada={self.precio_entrada:.4f} SL={self.stop_loss:.4f} "
            f"TP={self.take_profit:.4f} vol={self.volumen} "
            f"riesgo=${self.riesgo_dolares:.0f} R/R={self.ratio_rr:.1f}:1)"
        )


class RiskManager:
    """
    Calcula el sizing completo para cada señal detectada.
    """

    def __init__(
        self,
        capital:          float = CAPITAL_DEFAULT,
        riesgo_pct:       float = RIESGO_POR_OP,
        stop_buffer_pct:  float = STOP_BUFFER_PCT,
        dist_sl_min_pct:  float = DIST_SL_MIN_PCT,
        dist_sl_max_pct:  float = DIST_SL_MAX_PCT,
        max_posiciones:   int   = MAX_POSICIONES,
        max_drawdown_kill:float  = MAX_DRAWDOWN_KILL,
        min_ratio_rr:     float = MIN_RATIO_RR,
    ):
        self.capital           = capital
        self.riesgo_pct        = riesgo_pct
        self.stop_buffer_pct   = stop_buffer_pct
        self.dist_sl_min_pct   = dist_sl_min_pct
        self.dist_sl_max_pct   = dist_sl_max_pct
        self.max_posiciones    = max_posiciones
        self.max_drawdown_kill = max_drawdown_kill
        self.min_ratio_rr      = min_ratio_rr

        # Estado en tiempo real (actualizado desde el order manager)
        self._posiciones_abiertas: int   = 0
        self._equity_actual:       float = capital
        self._drawdown_actual:     float = 0.0

    # ══════════════════════════════════════════════════════════════
    #  MÉTODO PRINCIPAL
    # ══════════════════════════════════════════════════════════════

    def calcular_orden(
        self,
        señal:  Señal,
        df_d1:  pd.DataFrame,
    ) -> OrdenCalculada:
        """
        A partir de una Señal detectada y el DataFrame diario,
        calcula todos los parámetros de la orden y valida que
        cumple los criterios de riesgo.

        Args:
            señal:  Señal del pattern scanner
            df_d1:  DataFrame diario (para calcular SL con low/high)

        Returns:
            OrdenCalculada con todos los campos y valida=True/False
        """
        orden = OrdenCalculada(
            symbol         = señal.symbol,
            direccion      = señal.direccion,
            patron         = señal.patron,
            precio_entrada = señal.precio_entrada,
        )

        # ── 1. Verificar kill switch ──────────────────────────────
        if not self._verificar_kill_switch():
            orden.motivo_rechazo = (
                f"Kill switch activo: drawdown={self._drawdown_actual:.1%} "
                f"≥ {self.max_drawdown_kill:.1%}"
            )
            return orden

        # ── 2. Verificar límite de posiciones ─────────────────────
        if self._posiciones_abiertas >= self.max_posiciones:
            orden.motivo_rechazo = (
                f"Máx posiciones alcanzado: "
                f"{self._posiciones_abiertas}/{self.max_posiciones}"
            )
            return orden

        # ── 3. Calcular Stop Loss ─────────────────────────────────
        sl = self._calcular_stop_loss(señal, df_d1)
        if sl <= 0:
            orden.motivo_rechazo = "No se pudo calcular Stop Loss"
            return orden
        orden.stop_loss = sl

        # ── 4. Validar distancia al SL (6-9% del precio) ─────────
        dist_pct = abs(señal.precio_entrada - sl) / señal.precio_entrada
        orden.distancia_sl_pct = round(dist_pct * 100, 2)

        if dist_pct < self.dist_sl_min_pct:
            orden.motivo_rechazo = (
                f"SL demasiado ajustado: {dist_pct:.1%} < {self.dist_sl_min_pct:.1%}"
            )
            return orden

        if dist_pct > self.dist_sl_max_pct:
            orden.motivo_rechazo = (
                f"SL demasiado amplio: {dist_pct:.1%} > {self.dist_sl_max_pct:.1%}"
            )
            return orden

        # ── 5. Calcular tamaño de posición ────────────────────────
        riesgo_dolares = self._equity_actual * self.riesgo_pct
        distancia_abs  = abs(señal.precio_entrada - sl)
        volumen        = math.floor(riesgo_dolares / distancia_abs)

        if volumen <= 0:
            orden.motivo_rechazo = (
                f"Volumen calculado = 0 "
                f"(riesgo=${riesgo_dolares:.0f}, dist=${distancia_abs:.4f})"
            )
            return orden

        orden.volumen        = float(volumen)
        orden.riesgo_dolares = round(volumen * distancia_abs, 2)

        # ── 6. Take Profit ────────────────────────────────────────
        tp = señal.take_profit if señal.take_profit > 0 else self._calcular_take_profit(
            señal, sl, df_d1
        )
        
        # Validar que el TP esté del lado correcto
        if señal.direccion == "LONG" and tp <= señal.precio_entrada:
            orden.motivo_rechazo = f"TP inválido para LONG: {tp:.4f} <= {señal.precio_entrada:.4f}"
            return orden
        if señal.direccion == "SHORT" and tp >= señal.precio_entrada:
            orden.motivo_rechazo = f"TP inválido para SHORT: {tp:.4f} >= {señal.precio_entrada:.4f}"
            return orden

        orden.take_profit = tp

        # ── 7. Ratio R/R ──────────────────────────────────────────
        if tp > 0 and distancia_abs > 0:
            beneficio    = abs(tp - señal.precio_entrada)
            orden.ratio_rr = round(beneficio / distancia_abs, 2)

        # ── 8. Validar R/R mínimo ─────────────────────────────────
        if orden.ratio_rr > 0 and orden.ratio_rr < self.min_ratio_rr:
            orden.motivo_rechazo = (
                f"R/R insuficiente: {orden.ratio_rr:.1f}:1 < {self.min_ratio_rr:.1f}:1"
            )
            return orden

        orden.valida = True
        logger.info(
            f"Orden calculada: {orden}"
        )
        return orden

    def calcular_multiples(
        self,
        señales: list,
        df_d1_por_symbol: dict,
    ) -> list:
        """
        Calcula órdenes para una lista de señales.
        Solo devuelve las órdenes válidas.
        """
        ordenes_validas = []

        for señal in señales:
            df = df_d1_por_symbol.get(señal.symbol)
            if df is None:
                logger.warning(f"{señal.symbol}: sin datos D1 para risk manager")
                continue

            orden = self.calcular_orden(señal, df)

            if orden.valida:
                ordenes_validas.append(orden)
            else:
                logger.debug(
                    f"{señal.symbol} [{señal.patron}] rechazada: "
                    f"{orden.motivo_rechazo}"
                )

        logger.info(
            f"Risk Manager: {len(señales)} señales → "
            f"{len(ordenes_validas)} órdenes válidas"
        )
        return ordenes_validas

    # ══════════════════════════════════════════════════════════════
    #  ACTUALIZACIÓN DE ESTADO
    # ══════════════════════════════════════════════════════════════

    def actualizar_estado(
        self,
        posiciones_abiertas: int,
        equity_actual:       float,
    ) -> None:
        """
        Actualiza el estado del risk manager con datos
        del broker (llamar al inicio de cada ciclo).
        """
        self._posiciones_abiertas = posiciones_abiertas
        self._equity_actual       = equity_actual
        self._drawdown_actual     = max(
            0.0,
            (self.capital - equity_actual) / self.capital
        )

        if self._drawdown_actual > self.max_drawdown_kill * 0.7:
            logger.warning(
                f"Drawdown en {self._drawdown_actual:.1%} — "
                f"Kill switch al {self.max_drawdown_kill:.1%}"
            )

    def get_estado(self) -> dict:
        """Devuelve un resumen del estado actual de riesgo."""
        return {
            "capital_inicial":    self.capital,
            "equity_actual":      self._equity_actual,
            "drawdown_actual":    round(self._drawdown_actual, 4),
            "drawdown_pct":       f"{self._drawdown_actual:.1%}",
            "posiciones_abiertas":self._posiciones_abiertas,
            "max_posiciones":     self.max_posiciones,
            "riesgo_por_op_$":    round(self._equity_actual * self.riesgo_pct, 2),
            "kill_switch_activo": not self._verificar_kill_switch(),
            "riesgo_total_abierto_$": round(
                self._posiciones_abiertas
                * self._equity_actual
                * self.riesgo_pct, 2
            ),
        }

    # ══════════════════════════════════════════════════════════════
    #  HELPERS INTERNOS
    # ══════════════════════════════════════════════════════════════

    def _calcular_stop_loss(
        self,
        señal: Señal,
        df:    pd.DataFrame,
    ) -> float:
        """
        Regla del 1%:
          LONG:  ref = min(low_hoy, low_ayer)  → SL = ref × (1 - buffer)
          SHORT: ref = max(high_hoy, high_ayer) → SL = ref × (1 + buffer)

        Si la señal ya trae SL calculado, lo respetamos.
        """
        # Usar el SL de la señal si ya viene calculado
        if señal.stop_loss > 0:
            return señal.stop_loss

        if len(df) < 2:
            return 0.0

        ult  = df.iloc[-1]
        prev = df.iloc[-2]

        if señal.direccion == "LONG":
            ref = min(float(ult["low"]), float(prev["low"]))
            sl  = round(ref * (1 - self.stop_buffer_pct), 4)
        else:
            ref = max(float(ult["high"]), float(prev["high"]))
            sl  = round(ref * (1 + self.stop_buffer_pct), 4)

        return sl

    def _calcular_take_profit(
        self,
        señal: Señal,
        stop_loss: float,
        df: pd.DataFrame,
    ) -> float:
        """
        TP mínimo de 2:1 R/R si no hay referencia técnica.
        """
        riesgo = abs(señal.precio_entrada - stop_loss)
        if señal.direccion == "LONG":
            return round(señal.precio_entrada + riesgo * 2.5, 4)
        else:
            return round(señal.precio_entrada - riesgo * 2.5, 4)

    def _verificar_kill_switch(self) -> bool:
        """True si el bot PUEDE operar (drawdown dentro de límites)."""
        return self._drawdown_actual < self.max_drawdown_kill
