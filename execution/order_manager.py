"""
power4_bot/execution/order_manager.py
================================================
Envía órdenes pendientes a MetaTrader 5 y
gestiona las posiciones abiertas.

Tipos de orden usados:
  BUY_STOP_LIMIT  → Para señales LONG  (PC1, 1-2-3 alc, etc.)
  SELL_STOP_LIMIT → Para señales SHORT (PV1, 1-2-3 baj, etc.)

Flujo por señal:
  1. Construir request MT5
  2. Validar con order_check()
  3. Enviar con order_send()
  4. Confirmar resultado y registrar
================================================
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from execution.risk_manager import OrdenCalculada

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_DISPONIBLE = True
except ImportError:
    mt5 = None
    MT5_DISPONIBLE = False

# Magic number único del bot (identifica todas las órdenes)
MAGIC_NUMBER  = 20240001
COMMENT_PREFIX = "P4"   # Prefijo en comentario de orden
LIMIT_BUFFER_PCT = 0.003   # 0.3% de margen entre stop y limit


@dataclass
class ResultadoOrden:
    """Resultado del envío de una orden a MT5."""
    orden:       OrdenCalculada
    enviada:     bool  = False
    ticket:      int   = 0          # Ticket MT5 si fue aceptada
    retcode:     int   = 0          # Código de retorno MT5
    motivo:      str   = ""
    precio_real: float = 0.0        # Precio de ejecución real
    volumen_real:float = 0.0        # Volumen final normalizado (Lotes/Acciones)

    def __repr__(self):
        if self.enviada:
            return (
                f"ResultadoOrden(✅ ticket={self.ticket} "
                f"{self.orden.symbol} {self.orden.direccion} "
                f"@ {self.precio_real:.4f})"
            )
        return (
            f"ResultadoOrden(❌ {self.orden.symbol} "
            f"[{self.orden.patron}] — {self.motivo})"
        )


class OrderManager:
    """
    Gestiona el envío y seguimiento de órdenes en MT5.
    En modo paper (simulación) registra las órdenes sin enviarlas.
    """

    def __init__(self, modo: str = "paper", magic: int = MAGIC_NUMBER):
        self.modo  = modo     # "paper" | "live"
        self.magic = magic
        self._ordenes_enviadas: List[ResultadoOrden] = []

    # ══════════════════════════════════════════════════════════════
    #  ENVÍO DE ÓRDENES
    # ══════════════════════════════════════════════════════════════

    def enviar_orden(self, orden: OrdenCalculada) -> ResultadoOrden:
        """
        Envía una única orden pendiente (BUY_STOP o SELL_STOP) a MT5.

        En modo paper: simula el envío y asigna ticket ficticio.
        En modo live:  envía la orden real y espera confirmación.
        """
        resultado = ResultadoOrden(orden=orden)

        if not orden.valida:
            resultado.motivo = f"Orden inválida: {orden.motivo_rechazo}"
            return resultado

        if self.modo == "paper":
            return self._simular_envio(orden)

        # ── Modo live: envío real ─────────────────────────────────
        return self._enviar_mt5(orden)

    def enviar_multiples(
        self,
        ordenes: List[OrdenCalculada],
    ) -> List[ResultadoOrden]:
        """Envía una lista de órdenes y devuelve todos los resultados."""
        resultados = []
        for orden in ordenes:
            resultado = self.enviar_orden(orden)
            resultados.append(resultado)
            # Pausa mínima entre órdenes para no saturar el broker
            if self.modo == "live":
                time.sleep(0.1)

        enviadas = sum(1 for r in resultados if r.enviada)
        logger.info(
            f"Order Manager: {len(ordenes)} órdenes procesadas | "
            f"{enviadas} enviadas correctamente"
        )
        return resultados

    # ══════════════════════════════════════════════════════════════
    #  GESTIÓN DE POSICIONES ABIERTAS
    # ══════════════════════════════════════════════════════════════

    def get_posiciones_abiertas(self) -> list:
        """
        Devuelve las posiciones abiertas del bot en MT5.
        Filtra por magic number para solo ver las del bot.
        """
        if not MT5_DISPONIBLE or self.modo == "paper":
            return self._posiciones_simuladas()

        posiciones = mt5.positions_get()
        if posiciones is None:
            logger.error(f"Error obteniendo posiciones: {mt5.last_error()}")
            return []

        # Filtrar solo las posiciones del bot
        bot_pos = [p for p in posiciones if p.magic == self.magic]

        resultado = []
        for p in bot_pos:
            resultado.append({
                "ticket":    p.ticket,
                "symbol":    p.symbol,
                "tipo":      "LONG" if p.type == 0 else "SHORT",
                "volumen":   p.volume,
                "entrada":   p.price_open,
                "sl":        p.sl,
                "tp":        p.tp,
                "pnl":       p.profit,
                "tiempo":    p.time,
                "comment":   p.comment,
            })

        return resultado

    def modificar_stop_loss(
        self,
        ticket:    int,
        nuevo_sl:  float,
        nuevo_tp:  float = 0.0,
    ) -> bool:
        """
        Modifica el SL (y opcionalmente el TP) de una posición abierta.
        Usado por el trailing stop automático (Fase 6).
        """
        if not MT5_DISPONIBLE or self.modo == "paper":
            logger.info(
                f"[PAPER] Modificar ticket={ticket}: SL={nuevo_sl:.4f} TP={nuevo_tp:.4f}"
            )
            return True

        posicion = mt5.positions_get(ticket=ticket)
        if not posicion:
            logger.error(f"Posición {ticket} no encontrada")
            return False

        pos = posicion[0]
        info = mt5.symbol_info(pos.symbol)
        if info is None: return False

        def normalizar(p, tick):
            return round(round(p / tick) * tick, info.digits)

        sl_final = normalizar(nuevo_sl, info.trade_tick_size)
        tp_final = normalizar(nuevo_tp if nuevo_tp > 0 else pos.tp, info.trade_tick_size)

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   pos.symbol,
            "position": ticket,
            "sl":       float(sl_final),
            "tp":       float(tp_final),
            "magic":    self.magic,
        }

        resultado = mt5.order_send(request)

        if resultado is None or resultado.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                f"Error modificando SL ticket={ticket}: "
                f"{resultado.retcode if resultado else 'None'}"
            )
            return False

        logger.info(
            f"[TRADE] SL modificado: ticket={ticket} "
            f"SL={nuevo_sl:.4f} TP={nuevo_tp:.4f}"
        )
        return True

    def cerrar_posicion(self, ticket: int) -> bool:
        """Cierra una posición por ticket (para el kill switch)."""
        if not MT5_DISPONIBLE or self.modo == "paper":
            logger.info(f"[PAPER] Cerrar posición ticket={ticket}")
            return True

        posicion = mt5.positions_get(ticket=ticket)
        if not posicion:
            return False

        pos    = posicion[0]
        tipo   = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        precio = mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 \
                 else mt5.symbol_info_tick(pos.symbol).ask

        request = {
            "action":   mt5.TRADE_ACTION_DEAL,
            "symbol":   pos.symbol,
            "volume":   pos.volume,
            "type":     tipo,
            "position": ticket,
            "price":    precio,
            "magic":    self.magic,
            "comment":  f"{COMMENT_PREFIX}_CLOSE",
        }

        resultado = mt5.order_send(request)
        return resultado is not None and resultado.retcode == mt5.TRADE_RETCODE_DONE

    # ══════════════════════════════════════════════════════════════
    #  HELPERS INTERNOS
    # ══════════════════════════════════════════════════════════════

    def _enviar_mt5(self, orden: OrdenCalculada) -> ResultadoOrden:
        """Envía la orden real a MT5 con normalización de precios y volumen."""
        resultado = ResultadoOrden(orden=orden)

        if not MT5_DISPONIBLE:
            resultado.motivo = "MT5 no disponible"
            return resultado

        # 1. Obtener info detallada del símbolo para normalizar
        try:
            info = mt5.symbol_info(orden.symbol)
            if info is None:
                resultado.motivo = f"Símbolo {orden.symbol} no encontrado en MT5"
                return resultado

            # Asegurar que el símbolo esté visible en Market Watch
            if not info.visible:
                if not mt5.symbol_select(orden.symbol, True):
                    resultado.motivo = f"No se pudo activar el símbolo {orden.symbol}"
                    return resultado
                info = mt5.symbol_info(orden.symbol)
        except Exception as e:
            resultado.motivo = f"Error conectando con MT5: {str(e)}"
            return resultado

        # 2. Normalizar PRECIOS (alinear con trade_tick_size)
        def normalizar_precio(p, tick):
            return round(round(p / tick) * tick, info.digits)

        precio_entrada = normalizar_precio(orden.precio_entrada, info.trade_tick_size)
        sl = normalizar_precio(orden.stop_loss, info.trade_tick_size)
        tp = normalizar_precio(orden.take_profit, info.trade_tick_size) if orden.take_profit > 0 else 0.0

        # 3. Normalizar VOLUMEN (Lotes en MT5)
        # El bot calcula 'unidades' base (riesgo_usd / distancia_precio).
        # En MT5 se envían 'lotes'. 1 lote = info.trade_contract_size unidades.
        volumen_final = orden.volumen / max(1.0, info.trade_contract_size)
        
        # Ajustar al paso de volumen del broker (ej. 0.01 lotes)
        step = info.volume_step
        volumen_final = round(round(volumen_final / step) * step, 2)
        
        # Limitar a mínimos y máximos del broker
        if volumen_final < info.volume_min:
            # Si el riesgo es muy pequeño, subimos al mínimo operable para que la orden pase
            volumen_final = info.volume_min
        
        volumen_final = min(info.volume_max, volumen_final)

        # 4. Determinar tipo de orden (y ajustar si el precio ya se pasó)
        # Obtenemos precio actual para decidir si BUY_STOP o BUY_MARKET
        tick = mt5.symbol_info_tick(orden.symbol)
        if tick is None:
            resultado.motivo = "No se pudo obtener precio actual (tick)"
            return resultado

        # Umbral máximo de desviación aceptable: 0.5% del precio de entrada calculado.
        # Si el mercado superó ese nivel, la señal está caducada y el riesgo calculado
        # ya no corresponde al SL/TP original → rechazar la orden.
        MAX_DESVIACION_PCT = 0.005

        if orden.direccion == "LONG":
            if precio_entrada <= tick.ask:
                desviacion = (tick.ask - precio_entrada) / precio_entrada
                if desviacion > MAX_DESVIACION_PCT:
                    resultado.motivo = (
                        f"Señal caducada: precio ya superó la entrada en "
                        f"{desviacion:.2%} (máx {MAX_DESVIACION_PCT:.1%}). "
                        f"Entrada={precio_entrada:.4f} Ask={tick.ask:.4f}"
                    )
                    logger.warning(
                        f"[{orden.symbol}] Orden LONG rechazada — señal caducada. "
                        f"{resultado.motivo}"
                    )
                    return resultado
                # Ajuste menor (<0.5%): solo mover al mínimo técnico del broker
                min_dist = max(info.trade_stops_level, 30) * info.trade_tick_size
                precio_entrada_ajustado = normalizar_precio(tick.ask + min_dist, info.trade_tick_size)
                logger.warning(
                    f"[{orden.symbol}] Entrada LONG ajustada por tick: "
                    f"{precio_entrada:.4f} → {precio_entrada_ajustado:.4f} "
                    f"(desviación {desviacion:.3%})"
                )
                precio_entrada = precio_entrada_ajustado
            tipo_orden = mt5.ORDER_TYPE_BUY_STOP_LIMIT
            precio_limit = normalizar_precio(precio_entrada * (1 + LIMIT_BUFFER_PCT), info.trade_tick_size)
        else:
            if precio_entrada >= tick.bid:
                desviacion = (precio_entrada - tick.bid) / precio_entrada
                if desviacion > MAX_DESVIACION_PCT:
                    resultado.motivo = (
                        f"Señal caducada: precio ya bajó de la entrada en "
                        f"{desviacion:.2%} (máx {MAX_DESVIACION_PCT:.1%}). "
                        f"Entrada={precio_entrada:.4f} Bid={tick.bid:.4f}"
                    )
                    logger.warning(
                        f"[{orden.symbol}] Orden SHORT rechazada — señal caducada. "
                        f"{resultado.motivo}"
                    )
                    return resultado
                min_dist = max(info.trade_stops_level, 30) * info.trade_tick_size
                precio_entrada_ajustado = normalizar_precio(tick.bid - min_dist, info.trade_tick_size)
                logger.warning(
                    f"[{orden.symbol}] Entrada SHORT ajustada por tick: "
                    f"{precio_entrada:.4f} → {precio_entrada_ajustado:.4f} "
                    f"(desviación {desviacion:.3%})"
                )
                precio_entrada = precio_entrada_ajustado
            tipo_orden = mt5.ORDER_TYPE_SELL_STOP_LIMIT
            precio_limit = normalizar_precio(precio_entrada * (1 - LIMIT_BUFFER_PCT), info.trade_tick_size)

        # 5. Detectar filling mode (FOK=1, IOC=2 o RETURN=3)
        # Algunos brokers no tienen las constantes expuestas en el módulo mt5 de Python (AttributeError)
        # Usamos los valores numéricos estándar de la API de MT5: 
        # SYMBOL_FILLING_FOK = 1, SYMBOL_FILLING_IOC = 2
        
        filling = mt5.ORDER_FILLING_RETURN
        if info.filling_mode == 1: # SYMBOL_FILLING_FOK
            filling = mt5.ORDER_FILLING_FOK
        elif info.filling_mode == 2: # SYMBOL_FILLING_IOC
            filling = mt5.ORDER_FILLING_IOC
        else:
            filling = mt5.ORDER_FILLING_RETURN

        request = {
            "action":        mt5.TRADE_ACTION_PENDING,
            "symbol":        orden.symbol,
            "volume":        float(volumen_final),
            "type":          tipo_orden,
            "price":         float(precio_entrada),
            "price_stoplimit": float(precio_limit),
            "sl":            float(sl),
            "tp":            float(tp),
            "magic":         self.magic,
            "comment":       f"{COMMENT_PREFIX}_{orden.patron}",
            "type_time":     mt5.ORDER_TIME_GTC,
            "type_filling":  filling,
        }

        # Validar
        try:
            check = mt5.order_check(request)
            if check is None or check.retcode != 0:
                retcode = check.retcode if check else -1
                msg = f"Check failed: {retcode}"
                if retcode == 10016: msg = "Stops inválidos (muy cerca del precio)"
                if retcode == 10014: msg = "Volumen inválido para este activo"
                if retcode == 10027: msg = "AutoTrading desactivado en MT5"
                
                resultado.motivo  = f"{msg} (raw:{retcode})"
                resultado.retcode = retcode
                return resultado

            # Enviar
            res = mt5.order_send(request)
            if res is None:
                resultado.motivo = "MT5 no respondió a order_send"
                return resultado
            
            resultado.retcode = res.retcode
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                resultado.enviada     = True
                resultado.ticket      = res.order
                resultado.precio_real = res.price
                resultado.volumen_real = res.volume
                resultado.motivo      = "OK"
                logger.info(
                    f"[TRADE] Orden enviada: {orden.symbol} {orden.direccion} "
                    f"[{orden.patron}] ticket={res.order} "
                    f"entrada={res.price:.4f} vol={res.volume}"
                )
            else:
                resultado.motivo = f"Error MT5: {res.comment} (code:{res.retcode})"
                logger.error(f"Orden rechazada: {orden.symbol} code:{res.retcode}")

        except Exception as e:
            resultado.motivo = f"Excepción en envío: {str(e)}"
            logger.exception("Error crítico en _enviar_mt5")

        self._ordenes_enviadas.append(resultado)
        return resultado

    def _simular_envio(self, orden: OrdenCalculada) -> ResultadoOrden:
        """Simula el envío en modo paper trading."""
        import random
        ticket = random.randint(100000, 999999)

        # Simulación de volumen para paper (detectar forex)
        vol_sim = orden.volumen
        if any(x in orden.symbol for x in ["USD","JPY","EUR","AUD"]):
            vol_sim = max(0.01, vol_sim / 100000.0)

        resultado = ResultadoOrden(
            orden        = orden,
            enviada      = True,
            ticket       = ticket,
            retcode      = 10009,   # TRADE_RETCODE_DONE simulado
            motivo       = "PAPER OK",
            precio_real  = orden.precio_entrada,
            volumen_real = vol_sim,
        )

        logger.info(
            f"[PAPER] Orden simulada: {orden.symbol} {orden.direccion} "
            f"[{orden.patron}] ticket={ticket} "
            f"entrada={orden.precio_entrada:.4f} "
            f"SL={orden.stop_loss:.4f} TP={orden.take_profit:.4f} "
            f"vol={orden.volumen} riesgo=${orden.riesgo_dolares:.0f} "
            f"modo: STOP_LIMIT"
        )

        self._ordenes_enviadas.append(resultado)
        return resultado

    def _posiciones_simuladas(self) -> list:
        """Posiciones ficticias para modo paper."""
        return []

    def get_historial(self) -> List[ResultadoOrden]:
        """Devuelve el historial de órdenes enviadas en esta sesión."""
        return self._ordenes_enviadas
