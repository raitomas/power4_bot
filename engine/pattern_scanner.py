"""
power4_bot/engine/pattern_scanner.py
================================================
Orquestador de patrones.

Recibe un activo con sus DataFrames ya enriquecidos
y lanza todos los detectores aplicables según la
etapa y dirección confirmadas.

Reglas de filtrado previo (TODAS deben cumplirse):
  1. Alineamiento temporal confirmado (W1+D1 = E2 o E4)
  2. Precio cerca de SMA20 diaria (<4%) — salvo patrones
     que operan en exhalación (FRC, VRI, VVI)
  3. Solo timeframe diario como trigger
================================================
"""

import logging
from typing import List, Optional
import pandas as pd

from engine.stage_classifier import ResultadoAlineamiento, Etapa
from engine.patterns.base import Señal, SEÑAL_VACIA
from engine.patterns.pc1_pv1 import PC1, PV1
from engine.patterns.patron_123 import (
    Patron123Alcista, Patron123Bajista,
    Patron1234Alcista, Patron1234Bajista,
)
from engine.patterns.acunamiento import AcunamientoAlcista, AcunamientoBajista
from engine.patterns.otros_patrones import (
    FalloRupturaBajista, FalloRupturaAlcista,
    VelaRojaIgnorada, VelaVerdeIgnorada,
    PRCA, PRCB,
    FalloPV1, FalloPC1,
    HuecoProAlcista, HuecoProBajista,
    BottomTailEncubierta, TopTailEncubierta,
)

logger = logging.getLogger(__name__)

# ── Registro de patrones por dirección ──────────────────────────
# Cada entrada: (clase_patron, requiere_cerca_sma20)
PATRONES_LONG = [
    (PC1,                  True),
    (Patron123Alcista,     True),
    (Patron1234Alcista,    True),
    (AcunamientoAlcista,   False), # Opera sobre la SMA20
    (PRCA,                 True),
    (FalloRupturaBajista,  False), # Opera en exhalación
    (VelaRojaIgnorada,     True),
    (FalloPV1,             True),
    (HuecoProAlcista,      False), # Excepción exhalación
    (BottomTailEncubierta, True),
]

PATRONES_SHORT = [
    (PV1,                  True),
    (Patron123Bajista,     True),
    (Patron1234Bajista,    True),
    (AcunamientoBajista,   False),
    (PRCB,                 True),
    (FalloRupturaAlcista,  False),
    (VelaVerdeIgnorada,    True),
    (FalloPC1,             True),
    (HuecoProBajista,      False), # Excepción exhalación
    (TopTailEncubierta,    True),
]


def escanear(
    alineamiento: ResultadoAlineamiento,
    df_d1:        pd.DataFrame,
) -> List[Señal]:
    """
    Ejecuta todos los detectores de patrones CONFIRMADOS para un activo.
    Solo devuelve señales donde el trigger ya ocurrió (modo_entrada='confirmado').
    """
    symbol    = alineamiento.symbol
    señales   = []

    if not alineamiento.alineado:
        logger.debug(f"{symbol}: no alineado, saltando scanner.")
        return []

    direccion  = alineamiento.direccion
    es_long    = direccion == "LONG"
    patrones_base = PATRONES_LONG if es_long else PATRONES_SHORT

    es_rango = alineamiento.etapa_d1 in [Etapa.ACUMULACION, Etapa.DISTRIBUCION]
    patrones = []
    for p, req in patrones_base:
        if es_rango:
            if "Acunamiento" in p.__name__:
                patrones.append((p, req))
        else:
            patrones.append((p, req))

    for ClasePatron, requiere_cerca in patrones:
        if requiere_cerca and not alineamiento.cerca_sma20:
            logger.debug(
                f"{symbol} [{ClasePatron.nombre}]: saltado "
                f"(dist_sma20={alineamiento.dist_sma20:.1f}% > 4%)"
            )
            continue

        detector = ClasePatron()
        señal    = detector.evaluar(df_d1, symbol=symbol)

        if señal.detectado:
            señales.append(señal)

    if señales:
        logger.info(
            f"{symbol}: {len(señales)} señal(es) confirmada(s): "
            f"{', '.join(s.patron for s in señales)}"
        )
    return señales


def escanear_condicional(
    alineamiento: ResultadoAlineamiento,
    df_d1:        pd.DataFrame,
) -> List[Señal]:
    """
    Ejecuta los detectores de PRE-PATRÓN para un activo.
    Devuelve señales con modo_entrada='condicional' (BUY_STOP / SELL_STOP).
    """
    symbol  = alineamiento.symbol
    señales = []

    if not alineamiento.alineado:
        return []

    es_long       = alineamiento.direccion == "LONG"
    patrones_base = PATRONES_LONG if es_long else PATRONES_SHORT
    es_rango      = alineamiento.etapa_d1 in [Etapa.ACUMULACION, Etapa.DISTRIBUCION]

    for ClasePatron, requiere_cerca in patrones_base:
        if es_rango and "Acunamiento" not in ClasePatron.__name__:
            continue
        if requiere_cerca and not alineamiento.cerca_sma20:
            continue

        detector = ClasePatron()
        señal    = detector.evaluar_prepatron(df_d1, symbol=symbol)

        if señal is not None and señal.detectado:
            señales.append(señal)

    if señales:
        logger.info(
            f"{symbol}: {len(señales)} pre-patrón(es) condicional(es): "
            f"{', '.join(s.patron for s in señales)}"
        )
    return señales


def escanear_watchlist(
    resultados_alineamiento: List[ResultadoAlineamiento],
    datos_d1: dict,
    include_condicionales: bool = False,
) -> List[Señal]:
    """
    Escanea todos los activos de la watchlist.

    Args:
        resultados_alineamiento: Lista de ResultadoAlineamiento
        datos_d1: {symbol: df_d1_enriquecido}
        include_condicionales: Si True, también incluye señales condicionales
                               (pre-patrón con BUY_STOP / SELL_STOP pendiente)

    Returns:
        Lista global de señales, ordenadas por R/R descendente.
        Las confirmadas aparecen antes que las condicionales.
    """
    confirmadas  = []
    condicionales = []

    for alin in resultados_alineamiento:
        symbol = alin.symbol
        df_d1  = datos_d1.get(symbol)

        if df_d1 is None:
            logger.warning(f"{symbol}: sin datos D1 para scanner.")
            continue

        señales = escanear(alin, df_d1)
        confirmadas.extend(señales)

        if include_condicionales:
            # Solo añadir pre-patrón si no hay ya señal confirmada del mismo patrón
            syms_confirmados = {s.patron for s in señales}
            for s in escanear_condicional(alin, df_d1):
                if s.patron.replace(" PRE", "") not in syms_confirmados:
                    condicionales.append(s)

    # Ordenar dentro de cada grupo por R/R
    confirmadas.sort(key=lambda s: s.ratio_rr, reverse=True)
    condicionales.sort(key=lambda s: s.ratio_rr, reverse=True)

    todas = confirmadas + condicionales

    logger.info(
        f"Scanner completo: {len(resultados_alineamiento)} activos | "
        f"{len(confirmadas)} confirmadas | {len(condicionales)} condicionales"
    )
    return todas




def imprimir_señales(señales: List[Señal]) -> None:
    """Imprime tabla de señales en consola."""
    if not señales:
        print("\n  Sin señales activas.\n")
        return

    cab = (
        f"  {'SYM':>6} │ {'PATRÓN':>10} │ {'DIR':>5} │ "
        f"{'ENTRADA':>10} │ {'STOP':>10} │ {'TP':>10} │ {'R/R':>6}"
    )
    sep = "─" * len(cab)
    print(f"\n{sep}\n{cab}\n{sep}")

    for s in señales:
        print(
            f"  {s.symbol:>6} │ {s.patron:>10} │ {s.direccion:>5} │ "
            f"${s.precio_entrada:>9.4f} │ ${s.stop_loss:>9.4f} │ "
            f"${s.take_profit:>9.4f} │ {s.ratio_rr:>5.1f}:1"
        )

    print(f"{sep}\n  Total: {len(señales)} señal(es)\n")
