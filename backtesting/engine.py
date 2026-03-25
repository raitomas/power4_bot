"""
power4_bot/backtesting/engine.py
================================================
Motor de Backtesting del Método Power 4.

Simula el bot vela a vela sobre datos históricos
sin enviar órdenes reales. Registra cada operación
y calcula métricas de rendimiento completas.

Flujo por cada vela:
  1. Calcular indicadores hasta esa vela
  2. Clasificar etapa (W1 + D1)
  3. Ejecutar scanner de patrones
  4. Calcular sizing con RiskManager
  5. Simular entrada/salida con gestión de stops
  6. Registrar resultado
================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
import pandas as pd
import numpy as np

from engine.indicators import calcular_indicadores
from engine.stage_classifier import verificar_alineamiento, Etapa
from engine.pattern_scanner import escanear
from execution.risk_manager import RiskManager

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """Representa una operación completa en el backtesting."""
    symbol:          str   = ""
    patron:          str   = ""
    direccion:       str   = ""
    fecha_entrada:   date  = None
    fecha_salida:    date  = None
    precio_entrada:  float = 0.0
    precio_salida:   float = 0.0
    stop_loss:       float = 0.0
    take_profit:     float = 0.0
    volumen:         float = 0.0
    pnl_dolares:     float = 0.0
    pnl_r:           float = 0.0      # Resultado en múltiplos de R
    ganadora:        bool  = False
    motivo_salida:   str   = ""       # "TP", "SL", "TRAILING", "FIN_DATOS"
    dias_en_posicion:int   = 0

    @property
    def riesgo_inicial(self) -> float:
        return abs(self.precio_entrada - self.stop_loss) * self.volumen


@dataclass
class ResultadoBacktest:
    """Métricas completas del backtesting."""
    symbol:           str         = ""
    trades:           list        = field(default_factory=list)
    # Métricas principales
    total_trades:     int         = 0
    ganadoras:        int         = 0
    perdedoras:       int         = 0
    win_rate:         float       = 0.0
    pnl_total:        float       = 0.0
    pnl_medio:        float       = 0.0
    r_medio:          float       = 0.0
    profit_factor:    float       = 0.0
    sharpe_ratio:     float       = 0.0
    max_drawdown_pct: float       = 0.0
    racha_perdedoras: int         = 0     # Máxima racha perdedora
    # Por patrón
    por_patron:       dict        = field(default_factory=dict)
    # Equity curve
    equity_curve:     list        = field(default_factory=list)

    def calcular(self, capital_inicial: float = 100_000.0) -> None:
        """Calcula todas las métricas a partir de la lista de trades."""
        if not self.trades:
            return

        self.total_trades = len(self.trades)
        self.ganadoras    = sum(1 for t in self.trades if t.ganadora)
        self.perdedoras   = self.total_trades - self.ganadoras
        self.win_rate     = round(self.ganadoras / self.total_trades, 3)
        self.pnl_total    = round(sum(t.pnl_dolares for t in self.trades), 2)
        self.pnl_medio    = round(self.pnl_total / self.total_trades, 2)
        self.r_medio      = round(
            sum(t.pnl_r for t in self.trades) / self.total_trades, 2
        )

        # Profit Factor = suma ganancias / suma pérdidas
        ganancias = sum(t.pnl_dolares for t in self.trades if t.ganadora)
        perdidas  = abs(sum(t.pnl_dolares for t in self.trades if not t.ganadora))
        self.profit_factor = round(ganancias / perdidas, 2) if perdidas > 0 else float("inf")

        # Equity curve y drawdown
        equity = capital_inicial
        pico   = capital_inicial
        max_dd = 0.0
        curva  = [capital_inicial]

        for t in self.trades:
            equity += t.pnl_dolares
            curva.append(round(equity, 2))
            if equity > pico:
                pico = equity
            dd = (pico - equity) / pico
            if dd > max_dd:
                max_dd = dd

        self.equity_curve     = curva
        self.max_drawdown_pct = round(max_dd, 4)

        # Sharpe Ratio (anualizado, asumiendo ~252 días de trading)
        returns = [t.pnl_dolares / capital_inicial for t in self.trades]
        if len(returns) > 1:
            media_r = np.mean(returns)
            std_r   = np.std(returns, ddof=1)
            factor  = np.sqrt(252 / max(1, (self.trades[-1].fecha_salida
                             - self.trades[0].fecha_entrada).days / len(returns)))
            self.sharpe_ratio = round(
                (media_r / std_r * factor) if std_r > 0 else 0.0, 2
            )

        # Racha máxima perdedora
        racha_actual = 0
        max_racha    = 0
        for t in self.trades:
            if not t.ganadora:
                racha_actual += 1
                max_racha = max(max_racha, racha_actual)
            else:
                racha_actual = 0
        self.racha_perdedoras = max_racha

        # Métricas por patrón
        patrones = set(t.patron for t in self.trades)
        for p in patrones:
            trades_p  = [t for t in self.trades if t.patron == p]
            ganadoras_p = sum(1 for t in trades_p if t.ganadora)
            self.por_patron[p] = {
                "trades":       len(trades_p),
                "win_rate":     round(ganadoras_p / len(trades_p), 3),
                "pnl":          round(sum(t.pnl_dolares for t in trades_p), 2),
                "r_medio":      round(sum(t.pnl_r for t in trades_p) / len(trades_p), 2),
            }


# ══════════════════════════════════════════════════════════════════
#  MOTOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Reproduce el comportamiento del bot vela a vela
    sobre datos históricos.
    """

    def __init__(
        self,
        capital        = 100_000.0,
        riesgo_pct     = 0.005,
        comision_pct   = 0.001,    # 0.1% por operación
        min_barras     = 250,      # Mínimo de barras para empezar
        dist_sl_min_pct= 0.04,
        dist_sl_max_pct= 0.15,
        min_ratio_rr   = 1.5,
    ):
        self.capital      = capital
        self.riesgo_pct   = riesgo_pct
        self.comision_pct = comision_pct
        self.min_barras   = min_barras
        self.rm = RiskManager(
            capital        = capital,
            riesgo_pct     = riesgo_pct,
            dist_sl_min_pct= dist_sl_min_pct,
            dist_sl_max_pct= dist_sl_max_pct,
            min_ratio_rr   = min_ratio_rr,
        )

    def ejecutar(
        self,
        symbol: str,
        df_d1:  pd.DataFrame,
        df_w1:  pd.DataFrame,
    ) -> ResultadoBacktest:
        """
        Ejecuta el backtest completo para un símbolo.

        Args:
            symbol: Nombre del activo
            df_d1:  DataFrame diario OHLC (sin indicadores)
            df_w1:  DataFrame semanal OHLC (sin indicadores)

        Returns:
            ResultadoBacktest con todas las métricas
        """
        resultado = ResultadoBacktest(symbol=symbol)
        trades    = []

        # Necesitamos al menos min_barras para tener SMA200 estable
        n_total = len(df_d1)
        if n_total < self.min_barras + 50:
            logger.warning(
                f"{symbol}: insuficientes barras ({n_total}). "
                f"Mínimo: {self.min_barras + 50}"
            )
            return resultado

        posicion_abierta: Optional[Trade] = None

        # Pre-calcular indicadores una sola vez para todo el histórico
        df_d1_ind = calcular_indicadores(df_d1)
        df_w1_ind = calcular_indicadores(df_w1)

        # ── Replay vela a vela ────────────────────────────────────
        for i in range(self.min_barras, n_total):
            # Subconjunto de datos "conocidos" hasta la vela i (sin lookahead)
            df_slice_d1 = df_d1_ind.iloc[:i + 1].copy()
            
            # REGLA ORO: Los últimos 3 cierres NO pueden tener swings confirmados 
            # (necesitan 3 velas a la derecha para validarse).
            df_slice_d1.iloc[-3:, df_slice_d1.columns.get_indexer(["swing_high", "swing_low"])] = False
            df_slice_d1.iloc[-3:, df_slice_d1.columns.get_indexer(["swing_high_price", "swing_low_price"])] = np.nan

            fecha_actual = df_slice_d1.index[-1]
            vela_actual  = df_slice_d1.iloc[-1]

            # ── Gestionar posición abierta ────────────────────────
            if posicion_abierta is not None:
                trade_cerrado = self._evaluar_cierre(
                    posicion_abierta, vela_actual, fecha_actual
                )
                if trade_cerrado is not None:
                    trades.append(trade_cerrado)
                    posicion_abierta = None
                    # Actualizar equity del risk manager
                    equity = self.capital + sum(t.pnl_dolares for t in trades)
                    self.rm.actualizar_estado(0, equity)
                    continue   # No abrir nueva posición el mismo día

            # ── Buscar nueva señal solo si no hay posición ────────
            if posicion_abierta is None:
                # Subconjunto semanal hasta la semana correspondiente
                df_w1_slice = df_w1_ind[df_w1_ind.index <= fecha_actual]
                if len(df_w1_slice) < 50:
                    continue

                # Alineamiento
                alin = verificar_alineamiento(symbol, df_w1_slice, df_slice_d1)
                if not alin.operable:
                    continue

                # Scanner de patrones
                señales = escanear(alin, df_slice_d1)
                if not señales:
                    continue

                # Tomar la señal con mayor R/R
                mejor_señal = max(señales, key=lambda s: s.ratio_rr)

                # Sizing
                orden = self.rm.calcular_orden(mejor_señal, df_slice_d1)
                if not orden.valida:
                    continue

                # Abrir posición simulada
                posicion_abierta = Trade(
                    symbol         = symbol,
                    patron         = mejor_señal.patron,
                    direccion      = mejor_señal.direccion,
                    fecha_entrada  = fecha_actual.date() if hasattr(fecha_actual, "date") else fecha_actual,
                    precio_entrada = float(vela_actual["close"]),   # Entrada al cierre
                    stop_loss      = orden.stop_loss,
                    take_profit    = orden.take_profit,
                    volumen        = orden.volumen,
                )

        # Cerrar posición pendiente al final de los datos
        if posicion_abierta is not None:
            ult_vela  = df_d1.iloc[-1]
            ult_fecha = df_d1.index[-1]
            trade_final = self._cerrar_forzado(posicion_abierta, ult_vela, ult_fecha)
            trades.append(trade_final)

        resultado.trades = trades
        resultado.calcular(self.capital)

        logger.info(
            f"Backtest {symbol}: {resultado.total_trades} trades | "
            f"WR={resultado.win_rate:.1%} | "
            f"PF={resultado.profit_factor:.2f} | "
            f"Sharpe={resultado.sharpe_ratio:.2f} | "
            f"MaxDD={resultado.max_drawdown_pct:.1%}"
        )
        return resultado

    def ejecutar_watchlist(
        self,
        datos: dict,
    ) -> dict:
        """
        Ejecuta el backtest para todos los activos.

        Args:
            datos: {symbol: {"D1": df_d1, "W1": df_w1}}

        Returns:
            {symbol: ResultadoBacktest}
        """
        resultados = {}
        for symbol, tfs in datos.items():
            logger.info(f"Backtesting {symbol}...")
            try:
                r = self.ejecutar(symbol, tfs["D1"], tfs["W1"])
                resultados[symbol] = r
            except Exception as e:
                logger.error(f"Error en backtest {symbol}: {e}")
        return resultados

    # ── Helpers ──────────────────────────────────────────────────

    def _evaluar_cierre(
        self,
        pos:    Trade,
        vela:   pd.Series,
        fecha:  pd.Timestamp,
    ) -> Optional[Trade]:
        """
        Evalúa si la posición se cierra en esta vela por:
        - Hit del Take Profit (high/low de la vela)
        - Hit del Stop Loss (low/high de la vela)
        """
        high  = float(vela["high"])
        low   = float(vela["low"])
        close = float(vela["close"])

        if pos.direccion == "LONG":
            # TP hit
            if high >= pos.take_profit:
                return self._cerrar(pos, pos.take_profit, fecha, "TP")
            # SL hit
            if low <= pos.stop_loss:
                return self._cerrar(pos, pos.stop_loss, fecha, "SL")
        else:  # SHORT
            if low <= pos.take_profit:
                return self._cerrar(pos, pos.take_profit, fecha, "TP")
            if high >= pos.stop_loss:
                return self._cerrar(pos, pos.stop_loss, fecha, "SL")

        return None   # Posición sigue abierta

    def _cerrar(
        self,
        pos:          Trade,
        precio_salida:float,
        fecha:        pd.Timestamp,
        motivo:       str,
    ) -> Trade:
        """Cierra la posición y calcula el P&L."""
        pos.fecha_salida  = fecha.date() if hasattr(fecha, "date") else fecha
        pos.precio_salida = precio_salida
        pos.motivo_salida = motivo
        pos.dias_en_posicion = (pos.fecha_salida - pos.fecha_entrada).days

        # P&L bruto
        if pos.direccion == "LONG":
            pnl_por_accion = precio_salida - pos.precio_entrada
        else:
            pnl_por_accion = pos.precio_entrada - precio_salida

        # Descontar comisión (entrada + salida)
        comision = pos.precio_entrada * self.comision_pct * 2 * pos.volumen

        pos.pnl_dolares = round(pnl_por_accion * pos.volumen - comision, 2)
        pos.ganadora    = pos.pnl_dolares > 0

        # R múltiple
        riesgo_por_accion = abs(pos.precio_entrada - pos.stop_loss)
        pos.pnl_r = round(
            pnl_por_accion / riesgo_por_accion, 2
        ) if riesgo_por_accion > 0 else 0.0

        return pos

    def _cerrar_forzado(
        self,
        pos:   Trade,
        vela:  pd.Series,
        fecha: pd.Timestamp,
    ) -> Trade:
        """Cierra al precio de cierre de la última vela disponible."""
        return self._cerrar(pos, float(vela["close"]), fecha, "FIN_DATOS")
