"""
power4_bot/dashboard/app.py
================================================
Dashboard Streamlit del Power 4 Trading Bot.

Vistas:
  1. 📊 Overview      — KPIs y estado global
  2. 🔍 Etapas        — Tabla de alineamiento W1+D1
  3. ⚡ Señales        — Patrones detectados hoy
  4. 💼 Posiciones     — P&L en tiempo real
  5. 📈 Backtesting   — Resultados históricos
  6. ⚙️  Configuración — Parámetros del bot

Ejecutar:
  streamlit run dashboard/app.py
================================================
"""

import sys
import os
import logging
from datetime import date, timedelta

import pandas as pd
import numpy as np

# Añadir raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    STREAMLIT_OK = True
except ImportError:
    STREAMLIT_OK = False

# ── Importaciones del bot ────────────────────────────────────────
from core.logging_config import configurar_logging
from core.mt5_connector import conectar, desconectar
from core.data_fetcher import descargar_ohlc
from core.symbols import get_active_symbols
from engine.indicators import calcular_indicadores
from engine.stage_classifier import analizar_watchlist, Etapa
from engine.pattern_scanner import escanear_watchlist
from execution.risk_manager import RiskManager
from execution.order_manager import OrderManager
from execution.autotrader import AutoTrader
from backtesting.engine import BacktestEngine
from backtesting.metrics import resumen_global, trades_a_dataframe, imprimir_reporte

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════

if STREAMLIT_OK:
    st.set_page_config(
        page_title   = "Power 4 Bot",
        page_icon    = "📊",
        layout       = "wide",
        initial_sidebar_state = "expanded",
    )

    # CSS personalizado
    st.markdown("""
    <style>
        .metric-card {
            background: #1e2330;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 16px 20px;
        }
        .badge-long  { color: #00e5a0; font-weight: 700; }
        .badge-short { color: #ff4757; font-weight: 700; }
        .badge-none  { color: #555b6e; }
        .etapa-2 { color: #00e5a0; }
        .etapa-4 { color: #ff4757; }
        .etapa-1, .etapa-3 { color: #f5c842; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  FUNCIONES DE CARGA DE DATOS (con caché)
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
#  ESTADO DEL ESCÁNER (session_state keys)
# ══════════════════════════════════════════════════════════════════
#  scan_phase   : 0=inactivo | 1=cribado W1 | 2=datos D1 | 3=análisis | 4=listo
#  scan_simbolos: lista completa a procesar
#  scan_pendientes_f1 : índice de siguiente símbolo a cribar (fase 1)
#  scan_pendientes_f2 : índice de siguiente símbolo a enriquecer (fase 2)
#  scan_w1_ok   : {sym: w1_indicadores} — pasaron el cribado W1
#  scan_datos   : {sym: {W1, D1, info}} — resultado final
#  scan_cancel  : bool — solicitud de cancelación
# ══════════════════════════════════════════════════════════════════

_SCAN_LOTE    = 4   # símbolos por rerun en cada fase
_SCAN_TIMEOUT = 10  # segundos máximo por símbolo


def _init_scan_state():
    """Inicializa las keys del escáner en session_state si no existen."""
    defaults = {
        "scan_phase":          0,
        "scan_simbolos":       [],
        "scan_pendientes_f1":  0,
        "scan_pendientes_f2":  0,
        "scan_w1_ok":          {},
        "scan_datos":          {},
        "scan_cancel":         False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def iniciar_escaneo(config: dict):
    """Arranca un nuevo escaneo desde cero."""
    if not conectar():
        st.error("❌ No se pudo conectar con MT5. Abre MetaTrader 5 primero.")
        return

    simbolos = get_active_symbols(config)
    
    # Filtrar por Tiers si se solicita "Sólo Tier A"
    if config.get("priorizar_tier_a"):
        simbolos = [s for s in simbolos if s.get("prioridad", 1) == 1]
        
    if not simbolos:
        st.warning("La watchlist está vacía.")
        return

    st.session_state.scan_simbolos      = simbolos
    st.session_state.scan_phase         = 1
    st.session_state.scan_pendientes_f1 = 0
    st.session_state.scan_pendientes_f2 = 0
    st.session_state.scan_w1_ok         = {}
    st.session_state.scan_datos         = {}
    st.session_state.scan_cancel        = False
    st.rerun()


def _procesar_simbolo_f1(activo: dict, timeout: int) -> dict | None:
    """
    Fase 1: descarga W1 y califica etapa.
    Devuelve {sym, w1_ind} si el símbolo pasa el cribado, None si no.
    """
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    from engine.stage_classifier import clasificar_etapa, Etapa

    sym = activo["symbol"]

    def _trabajo():
        try:
            w1_raw = descargar_ohlc(sym, "W1", n_barras=100)
            if w1_raw is None or len(w1_raw) < 50:
                return None
            w1_ind = calcular_indicadores(w1_raw)
            res    = clasificar_etapa(w1_ind)
            if res.etapa == Etapa.DESCONOCIDA:
                return None
            return {"sym": sym, "w1_ind": w1_ind, "info": activo}
        except Exception as e:
            logger.warning(f"[F1] Error {sym}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_trabajo)
        done, _ = wait([fut], timeout=timeout)
        if not done:
            logger.warning(f"[F1] Timeout {sym}")
            return None
        return fut.result()


def _procesar_simbolo_f2(sym: str, w1_ind, info: dict, timeout: int) -> dict | None:
    """
    Fase 2: descarga D1 + calcula indicadores.
    Devuelve {W1, D1, info} o None si falla / timeout.
    """
    from concurrent.futures import ThreadPoolExecutor, wait

    def _trabajo():
        try:
            d1_raw = descargar_ohlc(sym, "D1", n_barras=1510)
            if d1_raw is None or len(d1_raw) < 100:
                return None
            return {"W1": w1_ind, "D1": calcular_indicadores(d1_raw), "info": info}
        except Exception as e:
            logger.warning(f"[F2] Error {sym}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_trabajo)
        done, _ = wait([fut], timeout=timeout)
        if not done:
            logger.warning(f"[F2] Timeout {sym}")
            return None
        return fut.result()


def ejecutar_tick_escaner(timeout_sym: int = _SCAN_TIMEOUT) -> bool:
    """
    Ejecuta un 'tick' del escáner: procesa el lote siguiente y devuelve
    True si aún hay trabajo pendiente, False si ha terminado.

    Debe llamarse en cada rerun del dashboard mientras scan_phase in (1,2,3).
    """
    ss = st.session_state

    if ss.scan_cancel or ss.scan_phase == 0 or ss.scan_phase == 4:
        return False

    simbolos = ss.scan_simbolos
    total    = len(simbolos)

    # ── FASE 1: Cribado W1 ──────────────────────────────────────────
    if ss.scan_phase == 1:
        desde = ss.scan_pendientes_f1
        hasta = min(desde + _SCAN_LOTE, total)

        for activo in simbolos[desde:hasta]:
            resultado = _procesar_simbolo_f1(activo, timeout_sym)
            if resultado:
                ss.scan_w1_ok[resultado["sym"]] = {
                    "w1_ind": resultado["w1_ind"],
                    "info":   resultado["info"],
                }

        ss.scan_pendientes_f1 = hasta

        if hasta >= total:
            ss.scan_phase         = 2
            ss.scan_pendientes_f2 = 0
        return True

    # ── FASE 2: Datos D1 ────────────────────────────────────────────
    if ss.scan_phase == 2:
        syms_ok  = list(ss.scan_w1_ok.keys())
        total_f2 = len(syms_ok)

        if total_f2 == 0:
            ss.scan_phase = 4   # nada que enriquecer
            return False

        desde = ss.scan_pendientes_f2
        hasta = min(desde + _SCAN_LOTE, total_f2)

        for sym in syms_ok[desde:hasta]:
            entry  = ss.scan_w1_ok[sym]
            datos  = _procesar_simbolo_f2(sym, entry["w1_ind"], entry["info"], timeout_sym)
            if datos:
                ss.scan_datos[sym] = datos

        ss.scan_pendientes_f2 = hasta

        if hasta >= total_f2:
            ss.scan_phase = 3
        return True

    # ── FASE 3: Análisis instantáneo (señales) ──────────────────────
    if ss.scan_phase == 3:
        ss.scan_phase = 4   # marcamos como listo antes de calcular
        return False        # el render en main() hará el análisis con scan_datos

    return False


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚡ Power 4 Bot")
        st.markdown("---")

        modo = st.selectbox(
            "Modo de operación",
            ["📄 Paper Trading", "💰 Live Trading"],
            index=0,
        )
        capital = st.number_input(
            "Capital ($)", min_value=1000, max_value=10_000_000,
            value=100_000, step=1000
        )
        riesgo_pct = st.slider(
            "Riesgo por operación (%)", 0.1, 2.0, 0.5, 0.1
        )
        max_pos = st.slider("Máx posiciones", 1, 10, 5)

        st.markdown("---")
        st.markdown("⚙️ **Límites de Riesgo**")
        dist_sl_min = st.slider("Distancia SL Mínima (%)", 0.0, 10.0, 4.0, 0.5)
        dist_sl_max = st.slider("Distancia SL Máxima (%)", 5.0, 30.0, 15.0, 1.0)
        min_rr      = st.slider("R/R Mínimo (Beneficio/Riesgo)", 0.0, 3.0, 1.5, 0.1)

        st.markdown("---")
        st.markdown("🔍 **Descubrimiento**")
        desc_mode = st.selectbox(
            "Fuente de activos",
            ["📋 Watchlist (YAML)", "🖥️ Market Watch (MT5)"],
            index=1,
        )
        
        profundidad = st.selectbox(
            "Profundidad de escaneo",
            ["🚀 Sólo Tier A (194 activos)", "🐢 Completo (Tier A + B)"],
            index=0,
            help="Tier A incluye Forex, Índices, Metales y Top Stocks. Es mucho más rápido."
        )

        st.markdown("---")
        st.markdown("⚙️ **Escáner**")
        include_cond = st.toggle("Incluir Señales Condicionales", value=True, help="Detecta patrones en formación (pre-trigger) para poner órdenes pendientes.")
        timeout_sym = st.slider("Timeout por símbolo (s)", 3, 30, _SCAN_TIMEOUT, 1)

        ss = st.session_state
        fase_actual = ss.get("scan_phase", 0)
        escaneando  = fase_actual in (1, 2, 3)

        col_ini, col_can = st.columns(2)
        with col_ini:
            if st.button("🔄 Escanear", use_container_width=True, disabled=escaneando):
                # Guardamos config en state para que los ticks la usen
                ss["_scan_config"]      = {
                    "modo":        "paper" if "Paper" in modo else "live",
                    "capital":     capital,
                    "riesgo_pct":  riesgo_pct / 100,
                    "max_pos":     max_pos,
                    "dist_sl_min": dist_sl_min / 100,
                    "dist_sl_max": dist_sl_max / 100,
                    "min_rr":      min_rr,
                    "include_condicionales": include_cond,
                    "priorizar_tier_a": ("Tier A" in profundidad),
                    "discovery":   {"mode": "market_watch" if "Market Watch" in desc_mode else "watchlist"},
                }
                ss["_scan_timeout"] = timeout_sym
                iniciar_escaneo(ss["_scan_config"])

        with col_can:
            if st.button("🛑 Cancelar", use_container_width=True, disabled=not escaneando):
                ss.scan_cancel  = True
                ss.scan_phase   = 4   # marcar como listo con datos parciales
                st.rerun()

        # Estado del escáner
        if fase_actual == 0:
            st.caption("Pulsa **Escanear** para iniciar.")
        elif escaneando:
            total = len(ss.get("scan_simbolos", []))
            done_f1 = ss.get("scan_pendientes_f1", 0)
            done_f2 = ss.get("scan_pendientes_f2", 0)
            w1_ok   = len(ss.get("scan_w1_ok", {}))
            if fase_actual == 1:
                st.caption(f"🔍 Fase 1/2 — Cribando {done_f1}/{total}")
            else:
                st.caption(f"📥 Fase 2/2 — Enriqueciendo {done_f2}/{w1_ok}")
        elif fase_actual == 4:
            total_datos = len(ss.get("scan_datos", {}))
            cancelado   = ss.get("scan_cancel", False)
            label = "⚠️ Cancelado" if cancelado else "✅ Completado"
            st.caption(f"{label} — {total_datos} activo(s) procesados")

        return {
            "modo":        "paper" if "Paper" in modo else "live",
            "capital":     capital,
            "riesgo_pct":  riesgo_pct / 100,
            "max_pos":     max_pos,
            "dist_sl_min": dist_sl_min / 100,
            "dist_sl_max": dist_sl_max / 100,
            "min_rr":      min_rr,
            "include_condicionales": include_cond,
            "priorizar_tier_a": ("Tier A" in profundidad),
            "discovery":   {"mode": "market_watch" if "Market Watch" in desc_mode else "watchlist"},
            "timeout_sym": timeout_sym,
        }


# ══════════════════════════════════════════════════════════════════
#  VISTA 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════

def render_overview(analisis: list, señales: list):
    st.header("📊 Overview del Sistema")

    alin    = analisis

    operables = sum(1 for a in alin if a.operable)
    longs     = sum(1 for a in alin if a.alineado and a.direccion == "LONG")
    shorts    = sum(1 for a in alin if a.alineado and a.direccion == "SHORT")

    # KPIs principales
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🎯 Señales activas", len(señales))
    c2.metric("✅ Operables", operables)
    c3.metric("📈 LONG alineados", longs)
    c4.metric("📉 SHORT alineados", shorts)
    c5.metric("📋 Total activos", len(alin))

    st.markdown("---")

    # Resumen de señales
    col_señales, col_distrib = st.columns([2, 1])

    with col_señales:
        st.subheader("⚡ Señales detectadas")
        if not señales:
            st.info("Sin señales activas en este momento.")
        else:
            for s in señales[:8]:
                color    = "🟢" if s.direccion == "LONG" else "🔴"
                st.markdown(
                    f"{color} **{s.symbol}** — {s.patron} | "
                    f"Entrada: `${s.precio_entrada:.2f}` | "
                    f"SL: `${s.stop_loss:.2f}` | "
                    f"TP: `${s.take_profit:.2f}` | "
                    f"R/R: **{s.ratio_rr:.1f}:1**"
                )

    with col_distrib:
        st.subheader("Distribución de etapas")
        etapas_count = {
            "E1 Acumulación": sum(1 for a in alin if a.etapa_d1.value == 1),
            "E2 Alcista":     sum(1 for a in alin if a.etapa_d1.value == 2),
            "E3 Distribución":sum(1 for a in alin if a.etapa_d1.value == 3),
            "E4 Bajista":     sum(1 for a in alin if a.etapa_d1.value == 4),
        }
        fig = px.pie(
            values = list(etapas_count.values()),
            names  = list(etapas_count.keys()),
            color_discrete_sequence = ["#555b6e", "#00e5a0", "#f5c842", "#ff4757"],
            hole   = 0.4,
        )
        fig.update_layout(
            showlegend=True, height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  VISTA 2: TABLA DE ETAPAS
# ══════════════════════════════════════════════════════════════════

def render_etapas(analisis: list):
    st.header("🔍 Estado de Etapas — Watchlist completa")

    alin = analisis
    if not alin:
        st.warning("Sin datos de alineamiento disponibles.")
        return

    rows = []
    for a in alin:
        rows.append({
            "Símbolo":      a.symbol,
            "Etapa W1":     f"E{a.etapa_w1.value} {a.etapa_w1.name[:3]}",
            "Etapa D1":     f"E{a.etapa_d1.value} {a.etapa_d1.name[:3]}",
            "Alineamiento": ("✅ LONG" if a.direccion == "LONG"
                             else "✅ SHORT" if a.direccion == "SHORT"
                             else "❌ —"),
            "Dist SMA20%":  f"{a.dist_sma20:.1f}%",
            "Operable":     "⚡ SÍ" if a.operable else "—",
        })

    df_tabla = pd.DataFrame(rows)
    st.dataframe(
        df_tabla,
        use_container_width=True,
        height=450,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("""
    **Regla de Oro del alineamiento:**
    - 🟢 **LONG** = Semanal E2 + Diario E2 + Distancia SMA20 < 4%
    - 🔴 **SHORT** = Semanal E4 + Diario E4 + Distancia SMA20 < 4%
    """)


# ══════════════════════════════════════════════════════════════════
#  VISTA 3: SEÑALES DETALLADAS + EJECUCIÓN
# ══════════════════════════════════════════════════════════════════

def render_señales(señales: list, config: dict, datos_raw: dict):
    st.header("⚡ Señales Activas")
    modo    = config["modo"]  # "paper" | "live"

    if not señales:
        st.info("No hay señales activas. El mercado no presenta condiciones operables en este momento.")
        return

    # Banner modo activo
    if modo == "live":
        st.error("🔴 MODO LIVE — Las órdenes se enviarán a MetaTrader 5 REAL")
    else:
        st.info("📝 MODO PAPER — Las órdenes se simulan sin enviarlas al broker")

    # Historial de órdenes en sesión
    if "ordenes_ejecutadas" not in st.session_state:
        st.session_state.ordenes_ejecutadas = []

    rm = RiskManager(
        capital        = config["capital"],
        riesgo_pct     = config["riesgo_pct"],
        max_posiciones = config["max_pos"],
        dist_sl_min_pct= config["dist_sl_min"],
        dist_sl_max_pct= config["dist_sl_max"],
        min_ratio_rr   = config["min_rr"],
    )
    om = OrderManager(modo=modo)

    st.markdown(f"### {len(señales)} señal(es) detectadas")

    for i, s in enumerate(señales):
        es_cond = s.es_condicional()
        icon  = "⏳" if es_cond else ("📈" if s.direccion == "LONG" else "📉")
        color = "🔵" if es_cond else ("🟢" if s.direccion == "LONG" else "🔴")
        
        tipo_badge = " 💡 CONDICIONAL" if es_cond else " ✨ CONFIRMADA"
        badge = f"{color} {s.direccion}{tipo_badge}"

        fase_tag = f" | 🌀 {s.fase_mercado}" if s.fase_mercado else ""
        
        # Estilo visual diferente para condicionales
        with st.expander(
            f"{icon} **{s.symbol}** — {s.patron}{fase_tag} | {badge} | R/R {s.ratio_rr:.1f}:1",
            expanded=not es_cond, # Colapsar por defecto las condicionales si hay muchas
        ):
            # Métricas de la señal
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Dirección",  s.direccion)
            c2.metric("Nivel Activación" if es_cond else "Entrada", f"${s.precio_entrada:.4f}")
            c3.metric("Stop Loss",  f"${s.stop_loss:.4f}")
            c4.metric("Take Profit",f"${s.take_profit:.4f}")

            # Calcular sizing con RiskManager
            df_d1 = datos_raw.get(s.symbol, {}).get("D1")
            orden = rm.calcular_orden(s, df_d1) if df_d1 is not None else None

            if orden and orden.valida:
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Volumen",   int(orden.volumen))
                c6.metric("Riesgo $",  f"${orden.riesgo_dolares:.0f}")
                c7.metric("Dist SL",   f"{orden.distancia_sl_pct:.1f}%")
                c8.metric("R/R real",  f"{orden.ratio_rr:.1f}:1")
            elif orden:
                st.warning(f"⚠️ Risk Manager rechaza la orden: {orden.motivo_rechazo}")

            st.caption(f"📝 {s.razon}")
            if es_cond:
                st.info(f"ℹ️ Esta es una orden PENDIENTE. Solo se ejecutará si el precio toca {s.precio_entrada:.4f}.")
            
            st.markdown("---")

            # Botón de ejecución
            col_btn, col_estado = st.columns([1, 2])
            with col_btn:
                if es_cond:
                    label = f"⏳ Poner Orden Pendiente — {s.symbol}"
                else:
                    label = (
                        f"📤 Enviar a MT5 — {s.symbol}"
                        if modo == "live"
                        else f"📝 Paper Trade — {s.symbol}"
                    )
                
                ejecutar = st.button(
                    label,
                    key        = f"exec_{s.symbol}_{s.patron}_{i}",
                    type       = "primary" if not es_cond else "secondary",
                    disabled   = (orden is None or not orden.valida),
                    use_container_width = True,
                )

            with col_estado:
                # Mostrar última orden ejecutada para esta señal
                prev = [r for r in st.session_state.ordenes_ejecutadas
                        if r["symbol"] == s.symbol and r["patron"] == s.patron]
                for r in prev[-1:]:
                    if r["enviada"]:
                        # Detectar etiqueta según volumen (Forex suele ser < 1.0 en lotes)
                        tag = "lotes" if r["volumen_final"] < 1.0 or any(x in s.symbol for x in ["USD","JPY","EUR","AUD"]) else "acc."
                        st.success(
                            f"✅ Ticket #{r['ticket']} | "
                            f"{r['direccion']} {r['volumen_final']:.2f} {tag} @ ${r['entrada']:.4f} | "
                            f"SL ${r['sl']:.4f} | TP ${r['tp']:.4f}"
                        )
                    else:
                        st.error(f"❌ {r['motivo']}")

            # Procesar clic
            if ejecutar and orden and orden.valida:
                resultado = om.enviar_orden(orden)
                # El resultado ahora tiene el volumen real normalizado
                vol_real = resultado.volumen_real if hasattr(resultado, "volumen_real") else orden.volumen
                
                st.session_state.ordenes_ejecutadas.append({
                    "symbol":    s.symbol,
                    "patron":    s.patron,
                    "direccion": orden.direccion,
                    "volumen_final": vol_real,
                    "entrada":   resultado.precio_real if resultado.enviada else orden.precio_entrada,
                    "sl":        orden.stop_loss,
                    "tp":        orden.take_profit,
                    "ticket":    resultado.ticket,
                    "enviada":   resultado.enviada,
                    "motivo":    resultado.motivo,
                })
                st.rerun()

    # Historial de órdenes de esta sesión
    if st.session_state.ordenes_ejecutadas:
        st.markdown("---")
        st.subheader("📋 Historial de órdenes — sesión actual")
        df_hist = pd.DataFrame(st.session_state.ordenes_ejecutadas)
        df_hist["estado"] = df_hist["enviada"].map({True: "✅ OK", False: "❌ Error"})
        st.dataframe(
            df_hist[[
                        "symbol","patron","direccion","volumen_final",
                     "entrada","sl","tp","ticket","estado"]],
            use_container_width=True,
            hide_index=True,
        )
        if st.button("🗑️ Limpiar historial"):
            st.session_state.ordenes_ejecutadas = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════
#  VISTA 4: BACKTESTING
# ══════════════════════════════════════════════════════════════════

def render_backtesting(datos_raw: dict, config: dict):
    st.header("📈 Motor de Backtesting")

    col_config, col_exec = st.columns([1, 2])

    with col_config:
        st.subheader("Configuración")
        symbol_bt = st.selectbox(
            "Activo a testear",
            list(datos_raw.keys()),
            index=0,
        )
        capital_bt = st.number_input(
            "Capital inicial ($)", value=config["capital"],
            min_value=1000
        )
        riesgo_bt = st.slider(
            "Riesgo por operación (%)", 0.1, 2.0, 0.5, 0.1
        )
        periodo_anos = st.select_slider(
            "Período de backtest",
            options=[1, 2, 3, 5],
            value=2,
            format_func=lambda x: f"{x} año{'s' if x > 1 else ''}",
        )
        ejecutar  = st.button("▶️ Ejecutar Backtest", use_container_width=True)

    with col_exec:
        if ejecutar and symbol_bt in datos_raw:
            with st.spinner(f"Ejecutando backtest de {symbol_bt} ({periodo_anos} años)..."):
                # Recortar datos al período seleccionado
                df_d1_full = datos_raw[symbol_bt]["D1"]
                df_w1_full = datos_raw[symbol_bt]["W1"]

                # Calcular fecha de inicio del período
                from datetime import timedelta
                fecha_fin   = df_d1_full.index[-1]
                fecha_inicio = fecha_fin - pd.Timedelta(days=periodo_anos * 365)

                # Mantener 250 barras de calentamiento previas al periodo elegido
                df_d1_test = df_d1_full[df_d1_full.index >= (fecha_inicio - pd.Timedelta(days=365))]
                df_w1_test = df_w1_full

                engine = BacktestEngine(
                    capital        = capital_bt,
                    riesgo_pct     = riesgo_bt / 100,
                    dist_sl_min_pct= config["dist_sl_min"],
                    dist_sl_max_pct= config["dist_sl_max"],
                    min_ratio_rr   = config["min_rr"],
                )
                resultado = engine.ejecutar(
                    symbol_bt,
                    df_d1_test,
                    df_w1_test,
                )

            if resultado.total_trades == 0:
                st.warning("Sin trades en el período analizado.")
                return

            # KPIs
            st.subheader(f"Resultados — {symbol_bt}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Win Rate",      f"{resultado.win_rate:.1%}")
            c2.metric("Profit Factor", f"{resultado.profit_factor:.2f}")
            c3.metric("Sharpe Ratio",  f"{resultado.sharpe_ratio:.2f}")
            c4.metric("Max Drawdown",  f"{resultado.max_drawdown_pct:.1%}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Total Trades",  resultado.total_trades)
            c6.metric("P&L Total",     f"${resultado.pnl_total:,.0f}")
            c7.metric("R Medio",       f"{resultado.r_medio:.2f}R")
            c8.metric("Max Racha SL",  resultado.racha_perdedoras)

            # Equity Curve
            st.subheader("Equity Curve")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                y    = resultado.equity_curve,
                mode = "lines",
                line = dict(color="#00e5a0", width=2),
                fill = "tozeroy",
                fillcolor = "rgba(0,229,160,0.05)",
                name = "Equity",
            ))
            fig_eq.update_layout(
                height       = 280,
                paper_bgcolor= "rgba(0,0,0,0)",
                plot_bgcolor = "rgba(0,0,0,0)",
                showlegend   = False,
                xaxis = dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                yaxis = dict(
                    showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                    tickprefix="$", tickformat=",.0f"
                ),
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # Win Rate por patrón
            if resultado.por_patron:
                st.subheader("Rendimiento por patrón")
                df_pat = pd.DataFrame([
                    {
                        "Patrón":   p,
                        "Trades":   m["trades"],
                        "Win Rate": f"{m['win_rate']:.1%}",
                        "P&L":      f"${m['pnl']:,.0f}",
                        "R Medio":  f"{m['r_medio']:.2f}R",
                    }
                    for p, m in sorted(
                        resultado.por_patron.items(),
                        key=lambda x: -x[1]["pnl"]
                    )
                ])
                st.dataframe(df_pat, use_container_width=True, hide_index=True)

                # Gráfico de barras win rate
                fig_pat = px.bar(
                    x      = list(resultado.por_patron.keys()),
                    y      = [m["win_rate"] * 100 for m in resultado.por_patron.values()],
                    labels = {"x": "Patrón", "y": "Win Rate (%)"},
                    color  = [m["win_rate"] for m in resultado.por_patron.values()],
                    color_continuous_scale = ["#ff4757", "#f5c842", "#00e5a0"],
                )
                fig_pat.update_layout(
                    height=250, showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                )
                st.plotly_chart(fig_pat, use_container_width=True)

            # Log de trades
            st.subheader("Historial de operaciones")
            df_trades = trades_a_dataframe(resultado.trades)
            if not df_trades.empty:
                df_trades["resultado"] = df_trades["ganadora"].map(
                    {True: "✅ Ganadora", False: "❌ Perdedora"}
                )
                st.dataframe(
                    df_trades[[
                        "symbol","patron","direccion",
                        "fecha_entrada","fecha_salida",
                        "precio_entrada","precio_salida",
                        "pnl_dolares","pnl_r",
                        "motivo_salida","resultado",
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("Selecciona un activo y pulsa **▶️ Ejecutar Backtest**")


# ══════════════════════════════════════════════════════════════════
#  VISTA 5: AUTO-TRADING
# ══════════════════════════════════════════════════════════════════

def render_autotrading(config: dict):
    st.header("🤖 Modo Automático (Auto-Trader)")
    st.info("El bot escaneará el mercado periódicamente y ejecutará señales válidas automáticamente.")

    # Inicializar autotrader en session_state si no existe
    if "autotrader" not in st.session_state:
        st.session_state.autotrader = AutoTrader(config)

    at = st.session_state.autotrader

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if not at.running:
            if st.button("▶️ ACTIVAR AUTO-TRADING", type="primary", use_container_width=True):
                at.config.update(config) # Actualizar config con sliders actuales
                at.start()
                st.rerun()
        else:
            if st.button("🛑 DETENER AUTO-TRADING", use_container_width=True):
                at.stop()
                st.rerun()
    
    with c2:
        estado = "🟢 ACTIVO" if at.running else "⚪ INACTIVO"
        st.metric("Estado", estado)
    
    with c3:
        ult = at.ultimo_escaneo.strftime("%H:%M:%S") if at.ultimo_escaneo else "Nunca"
        st.metric("Último Escaneo", ult)

    st.markdown("---")
    
    # NUEVO: Tabla de Órdenes Pendientes (Condicionales)
    st.subheader("⏳ Órdenes Pendientes (Esperando Activación)")
    if at.pendientes_activas:
        rows_p = []
        import datetime
        ahora = datetime.datetime.now()
        
        for k, p in at.pendientes_activas.items():
            # Info ya viene en el dict p
            sym = p.get("symbol", "?")
            pat = p.get("patron", "?")
            s   = p.get("señal")
            
            expira_en = p.get("expira")
            dias_restantes = (expira_en - ahora).days if expira_en else "?"
            
            rows_p.append({
                "Activo": sym,
                "Patrón": pat,
                "Nivel Trigger": f"{s.nivel_activacion:.4f}" if s else "?",
                "Vela Origen": s.idx_vela if s else "?",
                "Expira en": f"{dias_restantes} días" if dias_restantes != "?" else "—",
                "Estado": "⏳ Pendiente"
            })
        
        df_pend = pd.DataFrame(rows_p)
        st.dataframe(df_pend, use_container_width=True, hide_index=True)
    else:
        st.info("No hay órdenes pendientes registradas.")

    st.markdown("---")
    
    # Historial de operaciones automáticas
    st.subheader("📋 Registro de Operaciones Automáticas")
    if at.operaciones_auto:
        df_auto = pd.DataFrame(at.operaciones_auto)
        df_auto["status"] = df_auto["enviada"].map({True: "✅ OK", False: "❌ Error"})
        st.dataframe(
            df_auto[["fecha", "symbol", "patron", "direccion", "volumen", "ticket", "status", "motivo"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No se han realizado operaciones automáticas todavía.")


# ══════════════════════════════════════════════════════════════════
#  VISTA 6: CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════

def render_config():
    st.header("⚙️ Configuración del Bot")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gestión de Riesgo")
        st.code("""
# config/settings.yaml
risk:
  capital: 100000.0
  riesgo_por_op: 0.005   # 0.5%
  max_posiciones: 5
  max_drawdown_kill: 0.08  # -8%
  stop_buffer_pct: 0.01
        """, language="yaml")

        st.subheader("Parámetros de Patrones")
        st.code("""
patrones:
  max_dist_sma20_pct: 4.0
  min_escalones_pc1_pv1: 3
  retroceso_max_123: 0.30
  lookback_swings: 20
        """, language="yaml")

    with col2:
        st.subheader("Estructura del Proyecto")
        st.code("""
power4_bot/
├── core/
│   ├── mt5_connector.py
│   ├── data_fetcher.py
│   └── symbols.py
├── engine/
│   ├── indicators.py
│   ├── stage_classifier.py
│   ├── pattern_scanner.py
│   └── patterns/
│       ├── pc1_pv1.py
│       ├── patron_123.py
│       ├── acunamiento.py
│       └── otros_patrones.py
├── execution/
│   ├── risk_manager.py
│   ├── order_manager.py
│   └── trailing_stop.py
├── backtesting/
│   ├── engine.py
│   └── metrics.py
└── dashboard/
    └── app.py
        """, language="bash")

    st.markdown("---")
    st.subheader("Estado del sistema")
    st.success("✅ Fases 1-8 implementadas y testeadas (144+ tests)")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Tests totales", "144+")
    col_b.metric("Patrones implementados", "10")
    col_c.metric("Cobertura código", "~85%")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def _render_progreso():
    """Muestra la barra de progreso mientras el escáner está activo."""
    ss     = st.session_state
    fase   = ss.get("scan_phase", 0)
    simbolos = ss.get("scan_simbolos", [])
    total  = len(simbolos)
    w1_ok  = len(ss.get("scan_w1_ok", {}))

    if total == 0:
        return

    # Calcular cuántos son Tier A vs B en la lista total
    n_tier_a = sum(1 for s in simbolos if s.get("prioridad", 1) == 1)
    n_tier_b = total - n_tier_a

    tier_info = f"  |  🟡 Tier A: {n_tier_a} (forex/índices/metales/top stocks)  ·  🔵 Tier B: {n_tier_b} (resto)"

    if fase == 1:
        done = ss.get("scan_pendientes_f1", 0)
        pct  = done / total if total else 0
        # Indicar si estamos procesando Tier A o Tier B
        sym_actual = simbolos[done - 1] if done > 0 else None
        tier_tag   = "🟡 Tier A" if (sym_actual and sym_actual.get("prioridad", 1) == 1) else "🔵 Tier B"
        label = (
            f"🔍 Fase 1/2: Cribado W1  [{tier_tag}] — {done}/{total} "
            f"({w1_ok} pasan el filtro){tier_info}"
        )
    elif fase == 2:
        done = ss.get("scan_pendientes_f2", 0)
        pct  = 0.5 + (done / w1_ok * 0.5) if w1_ok else 0.5
        syms_ok = list(ss.get("scan_w1_ok", {}).keys())
        sym_actual = syms_ok[done - 1] if done > 0 and done <= len(syms_ok) else None
        w1_info_map = ss.get("scan_w1_ok", {})
        prio_actual = w1_info_map.get(sym_actual, {}).get("info", {}).get("prioridad", 1) if sym_actual else 1
        tier_tag = "🟡 Tier A" if prio_actual == 1 else "🔵 Tier B"
        label = (
            f"📥 Fase 2/2: Datos D1  [{tier_tag}] — {done}/{w1_ok} "
            f"enriquecidos{tier_info}"
        )
    elif fase == 3:
        pct   = 0.99
        label = f"⚙️ Calculando análisis y señales...{tier_info}"
    else:
        return

    st.progress(min(pct, 1.0), text=label)


def main():
    configurar_logging()
    _init_scan_state()

    config = render_sidebar()

    ss    = st.session_state
    fase  = ss.get("scan_phase", 0)

    # ── Menú de navegación ──────────────────────────────────────────
    menu   = ["📊 Overview", "🔍 Etapas", "⚡ Señales", "🤖 Auto-Trading", "📈 Backtesting", "⚙️ Config"]
    choice = st.sidebar.radio("Navegación", menu)

    # ── Escáner en curso → procesar un lote y hacer rerun ───────────
    if fase in (1, 2, 3):
        _render_progreso()
        timeout_sym = ss.get("_scan_timeout", _SCAN_TIMEOUT)
        hay_mas = ejecutar_tick_escaner(timeout_sym)
        if hay_mas or fase in (1, 2):
            st.rerun()          # continuar con el próximo lote
        # Si hay_mas==False y fase==3 → el tick puso fase=4, caemos al bloque de abajo
        st.rerun()              # un último rerun para renderizar con fase==4

    # ── Sin datos aún ───────────────────────────────────────────────
    datos_raw = ss.get("scan_datos", {})

    if fase == 0:
        st.info(
            "👈 Pulsa **Escanear** en el panel lateral para iniciar el análisis.\n\n"
            "El escáner procesará los activos en 2 fases mostrando el progreso en tiempo real."
        )
        return

    if fase in (1, 2, 3):
        # Aún en progreso — mostrar resultados parciales si los hay
        _render_progreso()
        if datos_raw:
            st.info(f"Mostrando {len(datos_raw)} activo(s) ya procesados (escáner en curso...)")
        else:
            st.info("Escáner en progreso — espera que aparezcan los primeros resultados...")

    if not datos_raw:
        if fase == 4:
            st.warning(
                "No se obtuvieron datos. Comprueba que MT5 esté abierto "
                "y pulsa **Escanear** de nuevo."
            )
        return

    # ── Análisis con los datos disponibles ─────────────────────────
    analisis    = analizar_watchlist(datos_raw)
    datos_d1    = {sym: d["D1"] for sym, d in datos_raw.items()}
    
    include_cond = config.get("include_condicionales", True)
    señales_hoy = escanear_watchlist(analisis, datos_d1, include_condicionales=include_cond)

    # ── Render de la vista seleccionada ────────────────────────────
    if choice == "📊 Overview":
        render_overview(analisis, señales_hoy)
    elif choice == "🔍 Etapas":
        render_etapas(analisis)
    elif choice == "⚡ Señales":
        render_señales(señales_hoy, config, datos_raw)
    elif choice == "🤖 Auto-Trading":
        render_autotrading(config)
    elif choice == "📈 Backtesting":
        render_backtesting(datos_raw, config)
    else:
        render_config()


if __name__ == "__main__":
    if not STREAMLIT_OK:
        print("ERROR: Instala streamlit con: pip install streamlit plotly")
    else:
        main()
