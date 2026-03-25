"""
power4_bot/execution/autotrader.py
================================================
Motor de ejecución automática (Auto-Trading).
Escanea la watchlist, detecta señales en tiempo real
y las ejecuta automáticamente siguiendo las reglas 
de gestión de riesgo.
================================================
"""

import logging
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional

from core.symbols import get_active_symbols
from core.data_fetcher import descargar_ohlc
from engine.indicators import calcular_indicadores
from engine.stage_classifier import analizar_watchlist
from engine.pattern_scanner import escanear_watchlist
from execution.risk_manager import RiskManager
from execution.order_manager import OrderManager

logger = logging.getLogger("power4_bot.autotrader")

class AutoTrader:
    def __init__(self, config: dict):
        self.config = config
        self.running = False
        self.thread = None
        self.intervalo_scaneo_seg = config.get("intervalo_scaneo_min", 60) * 60

        # Gestores
        # NOTA: las claves deben coincidir exactamente con settings.yaml (sección "risk").
        # distancia_sl_min/max están en decimal (0.05 = 5%), igual que RiskManager las espera.
        self.rm = RiskManager(
            riesgo_pct        = config.get("riesgo_pct", 0.005),
            max_posiciones    = config.get("max_posiciones", 5),
            dist_sl_min_pct   = config.get("distancia_sl_min", 0.04),
            dist_sl_max_pct   = config.get("distancia_sl_max", 0.10),
            min_ratio_rr      = config.get("min_rr", 1.5)
        )
        self.om = OrderManager(modo=config.get("modo", "paper"))

        # Estado
        self.ultimo_escaneo: Optional[datetime] = None
        self.operaciones_auto: List[dict] = []

        # Órdenes pendientes activas: {ticket: {"señal": Señal, "expira": datetime}}
        self.pendientes_activas: dict = {}



    def start(self):
        """Inicia el bucle de autotrading en un hilo separado."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._bucle_principal, daemon=True)
        self.thread.start()
        logger.info("🤖 Modo Automático INICIADO")

    def stop(self):
        """Detiene el bucle."""
        self.running = False
        logger.info("🤖 Modo Automático DETENIDO")

    def _bucle_principal(self):
        """Bucle que ejecuta el ciclo cada cierto tiempo."""
        while self.running:
            try:
                self.ejecutar_ciclo()
            except Exception as e:
                logger.exception(f"Error crítico en ciclo de autotrading: {e}")
            
            # Esperar al siguiente intervalo o a que se detenga
            for _ in range(self.intervalo_scaneo_seg):
                if not self.running: break
                time.sleep(1)

    def ejecutar_ciclo(self):
        """Ejecuta una pasada completa: descarga → análisis → ejecución."""
        from concurrent.futures import ThreadPoolExecutor, wait as fut_wait
        from engine.stage_classifier import clasificar_etapa, Etapa

        logger.info("🔄 Iniciando escaneo automático...")
        self.ultimo_escaneo = datetime.now()

        simbolos = get_active_symbols(self.config)
        if not simbolos:
            return

        TIMEOUT_SYM = 15   # segundos máximo por símbolo
        datos_raw   = {}   # {sym: {W1, D1}}

        def _cargar(activo):
            sym = activo["symbol"]
            try:
                w1_raw = descargar_ohlc(sym, "W1", 100)
                if w1_raw is None or len(w1_raw) < 50:
                    return None
                w1_ind = calcular_indicadores(w1_raw)
                # Cribado rápido: descartar etapa desconocida
                if clasificar_etapa(w1_ind).etapa == Etapa.DESCONOCIDA:
                    return None
                d1_raw = descargar_ohlc(sym, "D1", 1000)   # mín 1000 barras
                if d1_raw is None or len(d1_raw) < 100:
                    return None
                return {"sym": sym, "W1": w1_ind, "D1": calcular_indicadores(d1_raw)}
            except Exception as e:
                logger.warning(f"[AutoTrader] Error {sym}: {e}")
                return None

        # Descarga con timeout por símbolo
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(_cargar, a): a for a in simbolos}
            done_set, _ = fut_wait(futures, timeout=TIMEOUT_SYM * len(simbolos))
            for fut in done_set:
                if not self.running:
                    break
                res = fut.result()
                if res:
                    datos_raw[res["sym"]] = {"W1": res["W1"], "D1": res["D1"]}

        if not datos_raw:
            logger.warning("[AutoTrader] Sin datos válidos en este ciclo.")
            return

        # 2. Análisis
        alineamientos = analizar_watchlist(datos_raw)
        datos_d1      = {sym: d["D1"] for sym, d in datos_raw.items()}

        # Incluir órdenes condicionales según configuración
        include_cond = self.config.get("include_condicionales", True)
        señales      = escanear_watchlist(alineamientos, datos_d1,
                                          include_condicionales=include_cond)

        # 3. Gestionar pendientes expiradas antes de añadir nuevas
        self._gestionar_pendientes()

        # 4. Ejecución
        for s in señales:
            if not self.running:
                break
            if self._ya_operado_hoy(s):
                continue

            df_s  = datos_d1.get(s.symbol)
            orden = self.rm.calcular_orden(s, df_s)

            if not (orden and orden.valida):
                continue

            if s.es_condicional():
                # ── Orden condicional: BUY_STOP / SELL_STOP ───────────
                if self._ya_tiene_pendiente(s.symbol, s.patron):
                    continue   # Ya hay una pendiente activa para este patrón
                logger.info(f"⏳ CONDICIONAL: {s.symbol} {s.patron} | "
                            f"Nivel={s.nivel_activacion:.4f}")
                resultado = self.om.enviar_orden(orden)
                if resultado.enviada:
                    from datetime import timedelta
                    expira = datetime.now() + timedelta(days=s.pendiente_expira)
                    self.pendientes_activas[resultado.ticket] = {
                        "señal":   s,
                        "expira":  expira,
                        "symbol":  s.symbol,
                        "patron":  s.patron,
                    }
                tipo_str = "CONDICIONAL"
            else:
                # ── Orden confirmada: entrada inmediata ───────────────
                logger.info(f"🚀 CONFIRMADA: {s.symbol} {s.patron} | Ejecutando...")
                resultado = self.om.enviar_orden(orden)
                tipo_str = "CONFIRMADA"

            self.operaciones_auto.append({
                "fecha":     datetime.now().strftime("%Y-%m-%d %H:%M"),
                "symbol":    s.symbol,
                "patron":    s.patron,
                "direccion": s.direccion,
                "tipo":      tipo_str,
                "enviada":   resultado.enviada,
                "ticket":    resultado.ticket,
                "motivo":    resultado.motivo,
                "volumen":   resultado.volumen_real if resultado.enviada else orden.volumen
            })

        logger.info("✓ Escaneo automático completado.")

    def _gestionar_pendientes(self):
        """
        Revisa las órdenes pendientes activas y cancela las que hayan expirado.
        """
        ahora   = datetime.now()
        a_borrar = []

        for ticket, info in self.pendientes_activas.items():
            if ahora >= info["expira"]:
                logger.info(
                    f"⌛ Pendiente expirada: ticket={ticket} "
                    f"{info['symbol']} {info['patron']} — cancelando..."
                )
                try:
                    if MT5_DISPONIBLE:
                        import MetaTrader5 as mt5
                        req = {
                            "action": mt5.TRADE_ACTION_REMOVE,
                            "order":  ticket,
                        }
                        mt5.order_send(req)
                except Exception as e:
                    logger.warning(f"Error cancelando ticket={ticket}: {e}")
                a_borrar.append(ticket)

        for t in a_borrar:
            del self.pendientes_activas[t]

    def _ya_tiene_pendiente(self, symbol: str, patron: str) -> bool:
        """Comprueba si ya existe una orden pendiente activa para este símbolo/patrón."""
        for info in self.pendientes_activas.values():
            if info["symbol"] == symbol and info["patron"] == patron:
                return True
        return False

    def _ya_operado_hoy(self, señal) -> bool:
        """Verifica si ya lanzamos una orden para este símbolo/patrón hoy."""
        hoy = datetime.now().date()
        for op in self.operaciones_auto:
            fecha_op = datetime.strptime(op["fecha"], "%Y-%m-%d %H:%M").date()
            if op["symbol"] == señal.symbol and op["patron"] == señal.patron and fecha_op == hoy:
                return True
        return False

