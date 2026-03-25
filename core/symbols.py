"""
power4_bot/core/symbols.py
================================================
Carga y valida la watchlist. Devuelve la lista
de símbolos activos y disponibles en el broker.
================================================
"""

import logging
import os
from typing import List, Dict

import yaml

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_DISPONIBLE = True
except ImportError:
    mt5 = None
    MT5_DISPONIBLE = False


def _cargar_watchlist(path: str = None) -> List[Dict]:
    """
    Lee el YAML de watchlist.
    Soporta dos formatos:
      · Nuevo (tier_a / tier_b): devuelve tier_a + tier_b en ese orden.
      · Antiguo (secciones planas): comportamiento original.

    Devuelve lista plana con el campo ``prioridad`` (1=tier_a, 2=tier_b).
    """
    if path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "config", "watchlist.yaml")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    simbolos: List[Dict] = []

    # ── Formato nuevo: tier_a / tier_b ──────────────────────────────
    if "tier_a" in data or "tier_b" in data:
        for tier_key, prioridad in [("tier_a", 1), ("tier_b", 2)]:
            for activo in data.get(tier_key, []):
                activo = dict(activo)           # copia para no mutar el YAML
                activo["prioridad"] = prioridad
                activo.setdefault("tipo", tier_key)
                simbolos.append(activo)
        logger.info(
            f"Watchlist (tiers) cargada: "
            f"{sum(1 for s in simbolos if s['prioridad']==1)} tier_a + "
            f"{sum(1 for s in simbolos if s['prioridad']==2)} tier_b"
        )
        return simbolos

    # ── Formato antiguo: secciones planas ───────────────────────────
    for categoria, activos in data.items():
        if isinstance(activos, list):
            for activo in activos:
                activo = dict(activo)
                activo["tipo"]      = categoria
                activo["prioridad"] = 1         # todo al mismo nivel
                simbolos.append(activo)

    logger.info(f"Watchlist (plana) cargada: {len(simbolos)} símbolos en {len(data)} categorías.")
    return simbolos




def validar_simbolo(symbol: str) -> bool:
    """
    Comprueba que el símbolo existe en el broker conectado
    y lo habilita si está desactivado.
    """
    if not MT5_DISPONIBLE:
        return True  # En simulación todos son válidos

    info = mt5.symbol_info(symbol)

    if info is None:
        logger.warning(f"Símbolo no encontrado en broker: {symbol}")
        return False

    # Activar símbolo en MarketWatch si no está visible
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"No se pudo activar {symbol} en MarketWatch.")
            return False
        logger.debug(f"Símbolo activado en MarketWatch: {symbol}")

    return True


def get_simbolos_validos(watchlist_path: str = None) -> List[Dict]:
    """
    Carga la watchlist y filtra solo los símbolos
    disponibles en el broker actual.

    Returns:
        Lista de dicts con symbol, name, category, tipo — solo los válidos.
    """
    todos = _cargar_watchlist(watchlist_path)
    validos = []
    invalidos = []

    for activo in todos:
        sym = activo["symbol"]
        if validar_simbolo(sym):
            validos.append(activo)
        else:
            invalidos.append(sym)

    if invalidos:
        logger.warning(
            f"Símbolos no disponibles en el broker ({len(invalidos)}): "
            f"{', '.join(invalidos)}"
        )

    logger.info(
        f"Símbolos válidos: {len(validos)} / {len(todos)} total."
    )
    return validos


# Palabras clave en el path de MT5 que indican Tier A automático
# (Forex, Metales, Índices, Commodities, ETFs principales)
_TIER_A_PATH_KEYWORDS = (
    "forex",  "currencies", "metals", "metal",
    "indices", "index",     "cfds",   "commodity", "commodities",
)
_TIER_A_ETF_SYMBOLS = {
    "VOO", "SPY", "QQQ", "TQQQ", "IWM", "GLD", "SLV",
    "EFA", "EEM", "VTI", "ARKK", "XLF", "XLE",
}
# Stocks de gran capitalización que van a Tier A junto a Forex/Índices
_TIER_A_TOP_STOCKS = {
    "AAPL", "MSFT", "NVDA", "TSLA", "META",
    "AMZN", "GOOGL", "GOOG", "NFLX", "JPM",
}


def _inferir_prioridad(symbol: str, path: str) -> int:
    """
    Devuelve 1 (Tier A) o 2 (Tier B) según el tipo de activo.

    Reglas automáticas:
      · Cualquier símbolo cuyo path MT5 contenga keywords de
        Forex / Metales / Índices / Commodities  → Tier A
      · ETFs conocidos                            → Tier A
      · Top stocks curados                        → Tier A
      · Resto (stocks secundarios, exóticos, crypto) → Tier B
    """
    path_lower = path.lower()
    if any(kw in path_lower for kw in _TIER_A_PATH_KEYWORDS):
        return 1
    if symbol in _TIER_A_ETF_SYMBOLS:
        return 1
    if symbol in _TIER_A_TOP_STOCKS:
        return 1
    return 2


def get_mt5_market_watch_symbols() -> List[Dict]:
    """
    Obtiene todos los símbolos VISIBLES en el Market Watch de MT5
    y les asigna prioridad automática según tipo de activo:
      · Forex, metales, índices, commodities → prioridad 1 (Tier A)
      · ETFs y top stocks conocidos          → prioridad 1 (Tier A)
      · Resto de stocks, crypto, exóticos    → prioridad 2 (Tier B)
    """
    if not MT5_DISPONIBLE:
        return []

    simbolos_mt5 = mt5.symbols_get()
    if simbolos_mt5 is None:
        logger.error(f"Error al obtener símbolos de MT5: {mt5.last_error()}")
        return []

    tier_a, tier_b = [], []
    for s in simbolos_mt5:
        if not s.visible:
            continue
        path     = s.path or ""
        tipo     = path.split("\\")[0] if "\\" in path else path
        prio     = _inferir_prioridad(s.name, path)
        entry    = {
            "symbol":    s.name,
            "name":      s.description if s.description else s.name,
            "category":  path,
            "tipo":      tipo,
            "prioridad": prio,
        }
        (tier_a if prio == 1 else tier_b).append(entry)

    logger.info(
        f"Market Watch: {len(tier_a)} Tier A "
        f"(forex/índices/metales/top stocks) + {len(tier_b)} Tier B"
    )
    return tier_a + tier_b          # Tier A siempre primero


def get_active_symbols(config: dict) -> List[Dict]:
    """
    Devuelve la lista de símbolos a escanear, SIEMPRE con Tier A primero.
    Funciona en modo 'market_watch' (clasificación automática por tipo MT5)
    y en modo 'watchlist' (clasificación por prioridad del YAML).
    """
    modo = config.get("discovery", {}).get("mode", "watchlist")

    if modo == "market_watch":
        return get_mt5_market_watch_symbols()

    # Modo watchlist: los símbolos ya traen campo 'prioridad' del YAML
    todos = get_simbolos_validos()
    tier_a = [s for s in todos if s.get("prioridad", 1) == 1]
    tier_b = [s for s in todos if s.get("prioridad", 2) == 2]
    logger.info(f"Watchlist ordenada: {len(tier_a)} Tier A + {len(tier_b)} Tier B")
    return tier_a + tier_b



def get_punto_decimal(symbol: str) -> float:
    """
    Devuelve el tamaño del punto (tick) para un símbolo.
    Necesario para calcular el valor monetario del stop loss.
    """
    if not MT5_DISPONIBLE:
        # Defaults razonables por categoría
        defaults = {
            "USD": 0.01, "JPY": 0.001,
            "BTC": 1.0,  "ETH": 0.01,
        }
        for k, v in defaults.items():
            if k in symbol:
                return v
        return 0.01

    info = mt5.symbol_info(symbol)
    return info.point if info else 0.01


def get_precio_actual(symbol: str) -> Dict:
    """
    Devuelve bid/ask actual de un símbolo.
    """
    if not MT5_DISPONIBLE:
        return {"bid": 100.0, "ask": 100.05, "spread": 0.05}

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"No se pudo obtener tick de {symbol}: {mt5.last_error()}")
        return {}

    return {
        "bid":    tick.bid,
        "ask":    tick.ask,
        "spread": round(tick.ask - tick.bid, 5),
        "time":   tick.time,
    }
