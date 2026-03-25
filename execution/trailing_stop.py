"""
power4_bot/execution/trailing_stop.py
================================================
Trailing Stop Automático — Fase 6

Se ejecuta diariamente al cierre de cada sesión.
Evalúa cada posición abierta y decide si mover
el Stop Loss según las reglas exactas del método.

Diagrama de flujo:
  Día 1-2: NO mover (salvo gap o vela excepcional)
  Día 3+:  Evaluar criterio "A Favor"
           LONG:  Close > Close_prev
                  AND Low > Low_prev
                  AND High > High_prev
           SHORT: imagen simétrica
  Si "A Favor":
           Ref = min(low[-2], low[-3])  para LONG
           Nuevo SL = Ref × 0.99
           Solo mover si nuevo SL > SL actual (antiretroceso)
================================================
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import pandas as pd

from execution.order_manager import OrderManager

logger = logging.getLogger(__name__)

STOP_BUFFER_PCT = 0.01   # 1% de buffer sobre/bajo la referencia


@dataclass
class EstadoPosicion:
    """
    Representación interna de una posición abierta
    para el cálculo del trailing stop.
    """
    ticket:        int
    symbol:        str
    direccion:     str    # "LONG" | "SHORT"
    fecha_entrada: date
    precio_entrada:float
    stop_loss:     float
    take_profit:   float
    volumen:       float
    patron:        str    = ""


@dataclass
class ResultadoTrailing:
    """Resultado de evaluar el trailing stop de una posición."""
    ticket:        int
    symbol:        str
    movido:        bool    = False
    sl_anterior:   float   = 0.0
    sl_nuevo:      float   = 0.0
    razon:         str     = ""
    dias_posicion: int     = 0

    def __repr__(self):
        if self.movido:
            return (
                f"Trailing(✅ {self.symbol} ticket={self.ticket} "
                f"SL: {self.sl_anterior:.4f} → {self.sl_nuevo:.4f} "
                f"[día {self.dias_posicion}])"
            )
        return (
            f"Trailing(— {self.symbol} ticket={self.ticket} "
            f"sin cambio: {self.razon})"
        )


class TrailingStopManager:
    """
    Evalúa y ejecuta el trailing stop para todas las
    posiciones abiertas del bot.
    """

    def __init__(self, order_manager: OrderManager):
        self.om = order_manager

    def gestionar_todas(
        self,
        posiciones:  list,
        datos_d1:    dict,
        fecha_hoy:   Optional[date] = None,
    ) -> list:
        """
        Evalúa el trailing stop de todas las posiciones abiertas.

        Args:
            posiciones: Lista de dicts con datos de posición (del OrderManager)
            datos_d1:   {symbol: DataFrame diario enriquecido}
            fecha_hoy:  Fecha actual (por defecto hoy)

        Returns:
            Lista de ResultadoTrailing
        """
        if fecha_hoy is None:
            fecha_hoy = date.today()

        resultados = []

        for pos_dict in posiciones:
            symbol = pos_dict.get("symbol", "")
            df     = datos_d1.get(symbol)

            if df is None:
                logger.warning(f"{symbol}: sin datos D1 para trailing stop")
                continue

            pos = EstadoPosicion(
                ticket         = pos_dict.get("ticket", 0),
                symbol         = symbol,
                direccion      = pos_dict.get("tipo", "LONG"),
                fecha_entrada  = _parse_fecha(pos_dict.get("tiempo", 0)),
                precio_entrada = pos_dict.get("entrada", 0.0),
                stop_loss      = pos_dict.get("sl", 0.0),
                take_profit    = pos_dict.get("tp", 0.0),
                volumen        = pos_dict.get("volumen", 0.0),
                patron         = pos_dict.get("comment", ""),
            )

            resultado = self.evaluar_posicion(pos, df, fecha_hoy)
            resultados.append(resultado)

            # Ejecutar movimiento si corresponde
            if resultado.movido:
                exito = self.om.modificar_stop_loss(
                    ticket   = pos.ticket,
                    nuevo_sl = resultado.sl_nuevo,
                    nuevo_tp = pos.take_profit,
                )
                if not exito:
                    resultado.movido = False
                    resultado.razon  = "Error al modificar SL en MT5"

        movidos = sum(1 for r in resultados if r.movido)
        logger.info(
            f"Trailing Stop: {len(resultados)} posiciones evaluadas | "
            f"{movidos} stops ajustados"
        )
        return resultados

    def evaluar_posicion(
        self,
        pos:        EstadoPosicion,
        df:         pd.DataFrame,
        fecha_hoy:  date,
    ) -> ResultadoTrailing:
        """
        Evalúa si se debe mover el stop de UNA posición.
        Implementa el diagrama de flujo exacto del método.
        """
        resultado = ResultadoTrailing(
            ticket       = pos.ticket,
            symbol       = pos.symbol,
            sl_anterior  = pos.stop_loss,
        )

        # ── Calcular días en posición ─────────────────────────────
        dias = (fecha_hoy - pos.fecha_entrada).days
        resultado.dias_posicion = dias

        # ── REGLA 1: Días 1-2 → NO mover ─────────────────────────
        if dias <= 2:
            # Excepción: gap a favor o vela sólida excepcional
            if self._hay_excepcion_dias_iniciales(df, pos):
                logger.debug(
                    f"{pos.symbol}: excepción días 1-2 "
                    f"(gap o vela excepcional)"
                )
            else:
                resultado.razon = f"Día {dias} ≤ 2: stop bloqueado"
                return resultado

        # ── REGLA 2: Día 3+, evaluar "A Favor" ───────────────────
        if not self._vela_a_favor(df, pos.direccion):
            resultado.razon = "Última vela NO fue 'A Favor'"
            return resultado

        # ── REGLA 3: Calcular nuevo SL ────────────────────────────
        nuevo_sl = self._calcular_nuevo_sl(df, pos)
        if nuevo_sl <= 0:
            resultado.razon = "No se pudo calcular nuevo SL"
            return resultado

        # ── REGLA 4: Antiretroceso ────────────────────────────────
        if pos.direccion == "LONG":
            # Para LONG: el nuevo SL debe ser MAYOR que el actual
            if nuevo_sl <= pos.stop_loss:
                resultado.razon = (
                    f"Antiretroceso: nuevo SL {nuevo_sl:.4f} "
                    f"≤ actual {pos.stop_loss:.4f}"
                )
                return resultado
        else:  # SHORT
            # Para SHORT: el nuevo SL debe ser MENOR que el actual
            if nuevo_sl >= pos.stop_loss:
                resultado.razon = (
                    f"Antiretroceso: nuevo SL {nuevo_sl:.4f} "
                    f"≥ actual {pos.stop_loss:.4f}"
                )
                return resultado

        # ── Mover el stop ─────────────────────────────────────────
        resultado.movido   = True
        resultado.sl_nuevo = nuevo_sl
        resultado.razon    = (
            f"Vela 'A Favor' en día {dias}. "
            f"SL: {pos.stop_loss:.4f} → {nuevo_sl:.4f}"
        )
        logger.info(
            f"[TRADE] {pos.symbol}: trailing stop activado. "
            f"{resultado.razon}"
        )
        return resultado

    # ══════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════

    def _vela_a_favor(
        self,
        df:        pd.DataFrame,
        direccion: str,
    ) -> bool:
        """
        Criterio "A Favor" del Método Power 4:
        LONG:  Close[n] > Close[n-1]  AND
               Low[n]   > Low[n-1]    AND
               High[n]  > High[n-1]
        SHORT: imagen simétrica (todo invertido)
        Ambos criterios deben cumplirse SIMULTÁNEAMENTE.
        """
        if len(df) < 2:
            return False

        actual   = df.iloc[-1]
        anterior = df.iloc[-2]

        if direccion == "LONG":
            criterio_precio = float(actual["close"]) > float(anterior["close"])
            criterio_rango  = (
                float(actual["low"])  > float(anterior["low"])  and
                float(actual["high"]) > float(anterior["high"])
            )
        else:  # SHORT
            criterio_precio = float(actual["close"]) < float(anterior["close"])
            criterio_rango  = (
                float(actual["high"]) < float(anterior["high"]) and
                float(actual["low"])  < float(anterior["low"])
            )

        return criterio_precio and criterio_rango

    def _calcular_nuevo_sl(
        self,
        df:  pd.DataFrame,
        pos: EstadoPosicion,
    ) -> float:
        """
        Cuando la vela va "A Favor", el nuevo SL se calcula
        mirando las DOS VELAS ANTERIORES a la actual:
          LONG:  ref = min(low[-2], low[-3])  → nuevo_SL = ref × 0.99
          SHORT: ref = max(high[-2], high[-3]) → nuevo_SL = ref × 1.01
        """
        v_actual = df.iloc[-1]
        v_prev   = df.iloc[-2]

        if pos.direccion == "LONG":
            ref      = min(float(v_actual["low"]), float(v_prev["low"]))
            nuevo_sl = round(ref * (1 - STOP_BUFFER_PCT), 4)
        else:
            ref      = max(float(v_actual["high"]), float(v_prev["high"]))
            nuevo_sl = round(ref * (1 + STOP_BUFFER_PCT), 4)

        return nuevo_sl

    def _hay_excepcion_dias_iniciales(
        self,
        df:  pd.DataFrame,
        pos: EstadoPosicion,
    ) -> bool:
        """
        Excepciones para mover el stop en días 1-2:
        1. Gap a favor (apertura muy por encima/debajo del cierre previo)
        2. Vela sólida excepcional (body > 2× ATR14)
        """
        if len(df) < 2:
            return False

        actual   = df.iloc[-1]
        anterior = df.iloc[-2]

        # Excepción 1: Gap a favor
        gap = abs(float(actual["open"]) - float(anterior["close"]))
        gap_pct = gap / float(anterior["close"])

        if pos.direccion == "LONG" and float(actual["open"]) > float(anterior["close"]):
            if gap_pct > 0.01:   # Gap > 1%
                return True
        elif pos.direccion == "SHORT" and float(actual["open"]) < float(anterior["close"]):
            if gap_pct > 0.01:
                return True

        # Excepción 2: Vela sólida excepcional (body > 2× ATR)
        atr14 = float(actual.get("atr14", 0) or 0)
        body  = abs(float(actual["close"]) - float(actual["open"]))
        if atr14 > 0 and body > 2 * atr14:
            return True

        return False


# ── Helper ────────────────────────────────────────────────────────

def _parse_fecha(tiempo) -> date:
    """Convierte timestamp MT5 o date a date."""
    if isinstance(tiempo, date):
        return tiempo
    if isinstance(tiempo, datetime):
        return tiempo.date()
    if isinstance(tiempo, (int, float)) and tiempo > 0:
        try:
            return datetime.fromtimestamp(tiempo).date()
        except Exception:
            pass
    return date.today()
