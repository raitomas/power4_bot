"""
power4_bot/engine/patterns/otros_patrones.py
================================================
Patrones adicionales del Método Power 4:

  • FRC Bajista / Alcista  — Fallo Ruptura Continuación
  • VRI                    — Vela Roja Ignorada
  • VVI                    — Vela Verde Ignorada
  • PRCA / PRCB            — Ruptura y Continuación
  • Fallo PV1 / PC1
================================================
"""

import logging
from typing import Optional, Union
import pandas as pd
import numpy as np

from engine.patterns.base import PatronBase, Señal, SEÑAL_VACIA

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  FALLO RUPTURA CONTINUACIÓN
# ══════════════════════════════════════════════════════════════════

class FalloRupturaBajista(PatronBase):
    """
    FRC Bajista → señal LONG (trampa bajista).
    Etapa 2: El precio rompe un soporte a la baja (Vela A)
    pero inmediatamente se revierte (Vela B, máx 2 días).
    El trigger es superar el máximo de la Vela A.
    """

    nombre             = "FRC_BAJ"
    direccion          = "LONG"
    MAX_DIST_SMA20_PCT = 0.0   # Opera en exhalación, sin límite Barrio Sésamo

    def _precondiciones(self, df: pd.DataFrame) -> tuple[bool, str]:
        if df is None or len(df) < 10:
            return False, "DataFrame insuficiente"
        if "sma20" not in df.columns:
            return False, "Indicadores no calculados"
        return True, ""

    def detectar(self, df: pd.DataFrame) -> Señal:
        n   = len(df)
        idx = n - 1

        # Necesitamos al menos 3 velas: A (ruptura), B (fallo), trigger
        for dias_atras in range(1, 3):   # Vela B máx 2 días después de A
            idx_a = idx - dias_atras - 1
            idx_b = idx - dias_atras
            idx_t = idx

            if idx_a < 5:
                break

            va = df.iloc[idx_a]
            vb = df.iloc[idx_b]
            vt = df.iloc[idx_t]

            # Vela A: bajista (intentó romper soporte)
            if float(va["close"]) >= float(va["open"]):
                continue

            # Vela A rompió un soporte (nuevo mínimo local)
            min_previo = df["low"].iloc[max(0, idx_a - 10) : idx_a].min()
            if float(va["low"]) >= min_previo:
                continue   # No hubo ruptura de soporte

            # Vela B: revierte al alza (cierra sobre el cierre de A o sobre SMA20)
            if float(vb["close"]) <= float(va["close"]):
                continue   # No hubo reversión

            # Trigger: precio actual supera el HIGH de la Vela A
            nivel_trigger = float(va["high"])
            if float(vt["close"]) <= nivel_trigger:
                continue

            precio_entrada = round(nivel_trigger * (1 + self.BUFFER_ENTRADA_PCT), 4)
            stop_loss      = round(float(va["low"]) * (1 - self.STOP_BUFFER_PCT), 4)

            return Señal(
                detectado      = True,
                precio_entrada = precio_entrada,
                stop_loss      = stop_loss,
                idx_vela       = idx_t,
                razon          = (
                    f"FRC_BAJ: Vela A (idx={idx_a}) rompió soporte {min_previo:.4f}, "
                    f"Vela B revirtió, trigger={nivel_trigger:.4f} superado."
                ),
                fase_mercado="Exhalación",
                datos_extra={
                    "idx_vela_a": idx_a, "idx_vela_b": idx_b,
                    "high_vela_a": nivel_trigger,
                    "low_vela_a": float(va["low"]),
                }
            )
        return SEÑAL_VACIA

    def detectar_prepatron(self, df: pd.DataFrame) -> Optional[Señal]:
        """Pre-patrón FRC Bajista: Vela A y B listas, trigger pendiente."""
        n = len(df); idx = n - 1
        if n < 5: return None
        for dias_atras in range(0, 2):
            idx_a, idx_b = idx - dias_atras - 1, idx - dias_atras
            if idx_a < 5: break
            va, vb = df.iloc[idx_a], df.iloc[idx_b]
            if float(va["close"]) >= float(va["open"]): continue
            min_previo = df["low"].iloc[max(0, idx_a - 10) : idx_a].min()
            if float(va["low"]) >= min_previo: continue
            if float(vb["close"]) <= float(va["close"]): continue
            nivel_trigger = float(va["high"])
            if float(df.iloc[idx]["close"]) < nivel_trigger:
                nv = round(nivel_trigger * (1 + self.BUFFER_ENTRADA_PCT), 4)
                return Señal(detectado=True, precio_entrada=nv, nivel_activacion=nv, modo_entrada="condicional",
                             idx_vela=idx, razon=f"FRC_BAJ PRE: Trigger en {nivel_trigger:.4f}.",
                             fase_mercado="Exhalación", datos_extra={"high_vela_a": nivel_trigger})
        return None


class FalloRupturaAlcista(PatronBase):
    """
    FRC Alcista → señal SHORT (trampa alcista).
    Etapa 4: El precio rompe una resistencia al alza (Vela A)
    pero falla en 1-2 días (Vela B) y se desploma.
    Trigger: pérdida del mínimo de la Vela A.
    """

    nombre             = "FRC_ALC"
    direccion          = "SHORT"
    MAX_DIST_SMA20_PCT = 0.0

    def _precondiciones(self, df: pd.DataFrame) -> tuple[bool, str]:
        if df is None or len(df) < 10:
            return False, "DataFrame insuficiente"
        if "sma20" not in df.columns:
            return False, "Indicadores no calculados"
        return True, ""

    def detectar(self, df: pd.DataFrame) -> Señal:
        n   = len(df)
        idx = n - 1

        for dias_atras in range(1, 3):
            idx_a = idx - dias_atras - 1
            idx_b = idx - dias_atras
            idx_t = idx

            if idx_a < 5:
                break

            va = df.iloc[idx_a]
            vb = df.iloc[idx_b]
            vt = df.iloc[idx_t]

            # Vela A: alcista (rompió resistencia)
            if float(va["close"]) <= float(va["open"]):
                continue

            max_previo = df["high"].iloc[max(0, idx_a - 10) : idx_a].max()
            if float(va["high"]) <= max_previo:
                continue

            # Vela B: falla (cierra bajo el cierre de A)
            if float(vb["close"]) >= float(va["close"]):
                continue

            # Trigger: pierde el LOW de Vela A
            nivel_trigger = float(va["low"])
            if float(vt["close"]) >= nivel_trigger:
                continue

            precio_entrada = round(nivel_trigger * (1 - self.BUFFER_ENTRADA_PCT), 4)
            stop_loss      = round(float(va["high"]) * (1 + self.STOP_BUFFER_PCT), 4)

            return Señal(
                detectado      = True,
                precio_entrada = precio_entrada,
                stop_loss      = stop_loss,
                idx_vela       = idx_t,
                razon          = (
                    f"FRC_ALC: Vela A (idx={idx_a}) falsa ruptura de {max_previo:.4f}, "
                    f"Vela B falló, trigger={nivel_trigger:.4f} perforado."
                ),
                fase_mercado="Exhalación",
                datos_extra={
                    "idx_vela_a": idx_a, "idx_vela_b": idx_b,
                    "low_vela_a": nivel_trigger,
                    "high_vela_a": float(va["high"]),
                }
            )
        return SEÑAL_VACIA

    def detectar_prepatron(self, df: pd.DataFrame) -> Optional[Señal]:
        """Pre-patrón FRC Alcista: Vela A y B listas, trigger pendiente."""
        n = len(df); idx = n - 1
        if n < 5: return None
        for dias_atras in range(0, 2):
            idx_a, idx_b = idx - dias_atras - 1, idx - dias_atras
            if idx_a < 5: break
            va, vb = df.iloc[idx_a], df.iloc[idx_b]
            if float(va["close"]) <= float(va["open"]): continue
            max_previo = df["high"].iloc[max(0, idx_a - 10) : idx_a].max()
            if float(va["high"]) <= max_previo: continue
            if float(vb["close"]) >= float(va["close"]): continue
            nivel_trigger = float(va["low"])
            if float(df.iloc[idx]["close"]) > nivel_trigger:
                nv = round(nivel_trigger * (1 - self.BUFFER_ENTRADA_PCT), 4)
                return Señal(detectado=True, precio_entrada=nv, nivel_activacion=nv, modo_entrada="condicional",
                             idx_vela=idx, razon=f"FRC_ALC PRE: Trigger en {nivel_trigger:.4f}.",
                             fase_mercado="Exhalación", datos_extra={"low_vela_a": nivel_trigger})
        return None


# ══════════════════════════════════════════════════════════════════
#  VELA ROJA IGNORADA (VRI) y VELA VERDE IGNORADA (VVI)
# ══════════════════════════════════════════════════════════════════

class VelaRojaIgnorada(PatronBase):
    """
    VRI — Vela Roja Ignorada (LONG, Etapa 2).
    En tendencia alcista fuerte aparece una vela roja
    pequeña de pausa que es superada de inmediato por
    la vela siguiente. El trigger es superar su máximo.
    """

    nombre    = "VRI"
    direccion = "LONG"

    def detectar(self, df: pd.DataFrame) -> Señal:
        n   = len(df)
        idx = n - 1
        if idx < 2: return SEÑAL_VACIA
        v_roja, v_trigger = df.iloc[idx - 1], df.iloc[idx]
        if float(v_roja["close"]) >= float(v_roja["open"]): return SEÑAL_VACIA
        body_roja = float(v_roja["open"]) - float(v_roja["close"])
        cuerpos = (df["close"].iloc[max(0, idx - 21) : idx - 1] - df["open"].iloc[max(0, idx - 21) : idx - 1]).abs()
        if len(cuerpos) > 0 and body_roja > np.percentile(cuerpos, 40): return SEÑAL_VACIA
        nivel_trigger = float(v_roja["high"])
        if float(v_trigger["close"]) <= nivel_trigger: return SEÑAL_VACIA
        precio_entrada = round(nivel_trigger * (1 + self.BUFFER_ENTRADA_PCT), 4)
        stop_loss      = round(float(v_roja["low"]) * (1 - self.STOP_BUFFER_PCT), 4)
        return Señal(detectado=True, precio_entrada=precio_entrada, stop_loss=stop_loss, idx_vela=idx,
                     razon=f"VRI: vela roja ignorada. Trigger={nivel_trigger:.4f} superado.",
                     fase_mercado="Exhalación", datos_extra={"high_vela_roja": nivel_trigger, "low_vela_roja": float(v_roja["low"])})

    def detectar_prepatron(self, df: pd.DataFrame) -> Optional[Señal]:
        """Pre-patrón VRI: Vela roja pequeña ayer o hoy, trigger pendiente."""
        n = len(df); idx = n - 1
        if idx < 1: return None
        v_roja = df.iloc[idx]
        if float(v_roja["close"]) >= float(v_roja["open"]): return None
        body_roja = float(v_roja["open"]) - float(v_roja["close"])
        cuerpos = (df["close"].iloc[max(0, idx - 20) : idx] - df["open"].iloc[max(0, idx - 20) : idx]).abs()
        if len(cuerpos) > 0 and body_roja > np.percentile(cuerpos, 40): return None
        nivel_trigger = float(v_roja["high"])
        nv = round(nivel_trigger * (1 + self.BUFFER_ENTRADA_PCT), 4)
        return Señal(detectado=True, precio_entrada=nv, nivel_activacion=nv, modo_entrada="condicional",
                     idx_vela=idx, razon=f"VRI PRE: Vela roja detectada. BUY_STOP en {nv:.4f}.",
                     fase_mercado="Exhalación", datos_extra={"high_vela_roja": nivel_trigger})


class VelaVerdeIgnorada(PatronBase):
    """
    VVI — Vela Verde Ignorada (SHORT, Etapa 4).
    En caída agresiva aparece una vela verde aislada
    que es perforada a la baja de inmediato.
    """

    nombre    = "VVI"
    direccion = "SHORT"

    def detectar(self, df: pd.DataFrame) -> Señal:
        n   = len(df)
        idx = n - 1
        if idx < 2: return SEÑAL_VACIA
        v_verde, v_trigger = df.iloc[idx - 1], df.iloc[idx]
        if float(v_verde["close"]) <= float(v_verde["open"]): return SEÑAL_VACIA
        body_verde = float(v_verde["close"]) - float(v_verde["open"])
        cuerpos = (df["close"].iloc[max(0, idx - 21) : idx - 1] - df["open"].iloc[max(0, idx - 21) : idx - 1]).abs()
        if len(cuerpos) > 0 and body_verde > np.percentile(cuerpos, 40): return SEÑAL_VACIA
        nivel_trigger = float(v_verde["low"])
        if float(v_trigger["close"]) >= nivel_trigger: return SEÑAL_VACIA
        precio_entrada = round(nivel_trigger * (1 - self.BUFFER_ENTRADA_PCT), 4)
        stop_loss      = round(float(v_verde["high"]) * (1 + self.STOP_BUFFER_PCT), 4)
        return Señal(detectado=True, precio_entrada=precio_entrada, stop_loss=stop_loss, idx_vela=idx,
                     razon=f"VVI: vela verde ignorada. Trigger={nivel_trigger:.4f} perforado.",
                     fase_mercado="Exhalación", datos_extra={"low_vela_verde": nivel_trigger, "high_vela_verde": float(v_verde["high"])})

    def detectar_prepatron(self, df: pd.DataFrame) -> Optional[Señal]:
        """Pre-patrón VVI: Vela verde pequeña detectada, trigger pendiente."""
        n = len(df); idx = n - 1
        if idx < 1: return None
        v_verde = df.iloc[idx]
        if float(v_verde["close"]) <= float(v_verde["open"]): return None
        body_verde = float(v_verde["close"]) - float(v_verde["open"])
        cuerpos = (df["close"].iloc[max(0, idx - 20) : idx] - df["open"].iloc[max(0, idx - 20) : idx]).abs()
        if len(cuerpos) > 0 and body_verde > np.percentile(cuerpos, 40): return None
        nivel_trigger = float(v_verde["low"])
        nv = round(nivel_trigger * (1 - self.BUFFER_ENTRADA_PCT), 4)
        return Señal(detectado=True, precio_entrada=nv, nivel_activacion=nv, modo_entrada="condicional",
                     idx_vela=idx, razon=f"VVI PRE: Vela verde detectada. SELL_STOP en {nv:.4f}.",
                     fase_mercado="Exhalación", datos_extra={"low_vela_verde": nivel_trigger})


# ══════════════════════════════════════════════════════════════════
#  PRCA / PRCB — Ruptura y Continuación
# ══════════════════════════════════════════════════════════════════

class PRCA(PatronBase):
    """
    PRCA — Patrón Ruptura y Continuación Alcista (LONG, Etapa 2).
    El precio rompe una zona de consolidación/resistencia
    al alza, preferiblemente cerca de la SMA20.
    """

    nombre    = "PRCA"
    direccion = "LONG"

    def detectar(self, df: pd.DataFrame) -> Señal:
        n = len(df); idx = n - 1

        # Zona de consolidación: barras de hace 3 a 35 barras atrás
        # (excluimos las últimas 2 para no confundir con el breakout actual)
        zona_start = max(0, idx - 35)
        zona_end   = max(0, idx - 2)
        ventana_conso = df.iloc[zona_start:zona_end]
        if len(ventana_conso) < 7: return SEÑAL_VACIA

        # Resistencia = máximo de la zona de consolidación
        resistencia = float(ventana_conso["high"].max())

        # Toques reales: barras que probaron la resistencia (dentro del 0.4%)
        banda_alta = resistencia * 0.996
        toques = int((ventana_conso["high"] >= banda_alta).sum())
        if toques < 2: return SEÑAL_VACIA

        # Consolidación mínima: al menos 6 barras con cierre en la zona
        # (precio pegado a la resistencia, no solo 1-2 picos aislados)
        zona_conso_pct = resistencia * 0.96   # dentro del 4% de la resistencia
        barras_en_zona = int((ventana_conso["close"] >= zona_conso_pct).sum())
        if barras_en_zona < 6: return SEÑAL_VACIA

        # Breakout: precio actual SUPERA la resistencia
        precio_actual = float(df.iloc[idx]["close"])
        if precio_actual <= resistencia: return SEÑAL_VACIA

        # No demasiado extendido: máximo 2.5% por encima de la resistencia
        extension = (precio_actual - resistencia) / resistencia
        if extension > 0.025: return SEÑAL_VACIA

        precio_entrada = round(resistencia * (1 + self.BUFFER_ENTRADA_PCT), 4)
        return Señal(
            detectado      = True,
            precio_entrada = precio_entrada,
            idx_vela       = idx,
            razon          = (
                f"PRCA: ruptura de resistencia {resistencia:.4f} "
                f"({toques} toques, {barras_en_zona} barras conso, "
                f"+{extension:.1%} sobre nivel)."
            ),
            datos_extra={
                "resistencia":  resistencia,
                "toques":       toques,
                "barras_conso": barras_en_zona,
            }
        )

    def detectar_prepatron(self, df: pd.DataFrame) -> Optional[Señal]:
        """Pre-patrón PRCA: consolidación formada bajo resistencia, breakout pendiente."""
        n = len(df); idx = n - 1

        zona_start = max(0, idx - 35)
        ventana_conso = df.iloc[zona_start : idx + 1]
        if len(ventana_conso) < 7: return None

        resistencia = float(ventana_conso["high"].max())
        banda_alta  = resistencia * 0.996
        toques = int((ventana_conso["high"] >= banda_alta).sum())
        if toques < 2: return None

        zona_conso_pct  = resistencia * 0.96
        barras_en_zona  = int((ventana_conso["close"] >= zona_conso_pct).sum())
        if barras_en_zona < 6: return None

        # Pre-patrón: precio AÚN por debajo de la resistencia
        precio_actual = float(df.iloc[idx]["close"])
        if precio_actual >= resistencia: return None

        nv = round(resistencia * (1 + self.BUFFER_ENTRADA_PCT), 4)
        return Señal(
            detectado        = True,
            precio_entrada   = nv,
            nivel_activacion = nv,
            modo_entrada     = "condicional",
            pendiente_expira = 3,
            idx_vela         = idx,
            razon            = (
                f"PRCA PRE: consolidación bajo {resistencia:.4f} "
                f"({toques} toques, {barras_en_zona} barras)."
            ),
            datos_extra={
                "resistencia":  resistencia,
                "toques":       toques,
                "barras_conso": barras_en_zona,
            }
        )


class PRCB(PatronBase):
    """
    PRCB — Patrón Ruptura y Continuación Bajista (SHORT, Etapa 4).
    Rotura a la baja de soporte para iniciar desplome.
    """

    nombre    = "PRCB"
    direccion = "SHORT"

    def detectar(self, df: pd.DataFrame) -> Señal:
        n = len(df); idx = n - 1

        zona_start = max(0, idx - 35)
        zona_end   = max(0, idx - 2)
        ventana_conso = df.iloc[zona_start:zona_end]
        if len(ventana_conso) < 7: return SEÑAL_VACIA

        # Soporte = mínimo de la zona de consolidación
        soporte = float(ventana_conso["low"].min())

        # Toques reales al soporte (dentro del 0.4%)
        banda_baja = soporte * 1.004
        toques = int((ventana_conso["low"] <= banda_baja).sum())
        if toques < 2: return SEÑAL_VACIA

        # Consolidación mínima
        zona_conso_pct = soporte * 1.04
        barras_en_zona = int((ventana_conso["close"] <= zona_conso_pct).sum())
        if barras_en_zona < 6: return SEÑAL_VACIA

        # Ruptura: precio actual PERFORA el soporte
        precio_actual = float(df.iloc[idx]["close"])
        if precio_actual >= soporte: return SEÑAL_VACIA

        # No demasiado extendido: máximo 2.5% por debajo
        extension = (soporte - precio_actual) / soporte
        if extension > 0.025: return SEÑAL_VACIA

        precio_entrada = round(soporte * (1 - self.BUFFER_ENTRADA_PCT), 4)
        return Señal(
            detectado      = True,
            precio_entrada = precio_entrada,
            idx_vela       = idx,
            razon          = (
                f"PRCB: ruptura de soporte {soporte:.4f} "
                f"({toques} toques, {barras_en_zona} barras conso, "
                f"-{extension:.1%} bajo nivel)."
            ),
            datos_extra={
                "soporte":      soporte,
                "toques":       toques,
                "barras_conso": barras_en_zona,
            }
        )

    def detectar_prepatron(self, df: pd.DataFrame) -> Optional[Señal]:
        """Pre-patrón PRCB: consolidación formada sobre soporte, ruptura pendiente."""
        n = len(df); idx = n - 1

        zona_start = max(0, idx - 35)
        ventana_conso = df.iloc[zona_start : idx + 1]
        if len(ventana_conso) < 7: return None

        soporte = float(ventana_conso["low"].min())
        banda_baja = soporte * 1.004
        toques = int((ventana_conso["low"] <= banda_baja).sum())
        if toques < 2: return None

        zona_conso_pct = soporte * 1.04
        barras_en_zona = int((ventana_conso["close"] <= zona_conso_pct).sum())
        if barras_en_zona < 6: return None

        precio_actual = float(df.iloc[idx]["close"])
        if precio_actual <= soporte: return None

        nv = round(soporte * (1 - self.BUFFER_ENTRADA_PCT), 4)
        return Señal(
            detectado        = True,
            precio_entrada   = nv,
            nivel_activacion = nv,
            modo_entrada     = "condicional",
            pendiente_expira = 3,
            idx_vela         = idx,
            razon            = (
                f"PRCB PRE: consolidación sobre {soporte:.4f} "
                f"({toques} toques, {barras_en_zona} barras)."
            ),
            datos_extra={
                "soporte":      soporte,
                "toques":       toques,
                "barras_conso": barras_en_zona,
            }
        )


# ══════════════════════════════════════════════════════════════════
#  NUEVOS PATRONES PLAYBOOK
# ══════════════════════════════════════════════════════════════════

class FalloPV1(PatronBase):
    """
    Fallo de PV1 → Señal Alcista (LONG, Etapa 2).
    Estructura de PV1 (mínimos crecientes) que falla y rompe al alza.
    """
    nombre    = "FALLO_PV1"
    direccion = "LONG"

    def detectar(self, df: pd.DataFrame) -> Señal:
        n = len(df); idx = n - 1
        if n < 5: return SEÑAL_VACIA
        lows = df["low"].iloc[idx-4:idx].values; highs = df["high"].iloc[idx-4:idx].values
        if not (lows[1] > lows[0] and lows[2] > lows[1]): return SEÑAL_VACIA
        max_estructura = max(highs)
        precio_actual = float(df.iloc[idx]["close"])
        if precio_actual > max_estructura:
            return Señal(detectado=True, precio_entrada=round(precio_actual, 4), idx_vela=idx,
                         razon=f"Fallo de PV1: estructura rota al alza.", fase_mercado="Inhalación")
        return SEÑAL_VACIA

    def detectar_prepatron(self, df: pd.DataFrame) -> Optional[Señal]:
        n = len(df); idx = n - 1
        if n < 5: return None
        lows = df["low"].iloc[idx-4:idx].values; highs = df["high"].iloc[idx-4:idx].values
        if not (lows[1] > lows[0] and lows[2] > lows[1]): return None
        max_estructura = max(highs)
        if float(df.iloc[idx]["close"]) < max_estructura:
            nv = round(max_estructura * (1 + self.BUFFER_ENTRADA_PCT), 4)
            return Señal(detectado=True, precio_entrada=nv, nivel_activacion=nv, modo_entrada="condicional",
                         idx_vela=idx, razon=f"FALLO_PV1 PRE: Máx en {max_estructura:.4f} pendiente.",
                         fase_mercado="Inhalación")
        return None


class FalloPC1(PatronBase):
    """
    Fallo de PC1 → Señal Bajista (SHORT, Etapa 4).
    Estructura de PC1 (máximos decrecientes) que falla y rompe a la baja.
    """
    nombre    = "FALLO_PC1"
    direccion = "SHORT"

    def detectar(self, df: pd.DataFrame) -> Señal:
        n = len(df); idx = n - 1
        if n < 5: return SEÑAL_VACIA
        highs = df["high"].iloc[idx-4:idx].values; lows = df["low"].iloc[idx-4:idx].values
        if not (highs[1] < highs[0] and highs[2] < highs[1]): return SEÑAL_VACIA
        min_estructura = min(lows)
        precio_actual = float(df.iloc[idx]["close"])
        if precio_actual < min_estructura:
            return Señal(detectado=True, precio_entrada=round(precio_actual, 4), idx_vela=idx,
                         razon=f"Fallo de PC1: estructura rota a la baja.", fase_mercado="Inhalación")
        return SEÑAL_VACIA

    def detectar_prepatron(self, df: pd.DataFrame) -> Optional[Señal]:
        n = len(df); idx = n - 1
        if n < 5: return None
        highs = df["high"].iloc[idx-4:idx].values; lows = df["low"].iloc[idx-4:idx].values
        if not (highs[1] < highs[0] and highs[2] < highs[1]): return None
        min_estructura = min(lows)
        if float(df.iloc[idx]["close"]) > min_estructura:
            nv = round(min_estructura * (1 - self.BUFFER_ENTRADA_PCT), 4)
            return Señal(detectado=True, precio_entrada=nv, nivel_activacion=nv, modo_entrada="condicional",
                         idx_vela=idx, razon=f"FALLO_PC1 PRE: Mín en {min_estructura:.4f} pendiente.",
                         fase_mercado="Inhalación")
        return None

class HuecoProAlcista(PatronBase):
    """
    Hueco PRO Alcista (LONG, Etapa 2).
    Gap significativo (>0.5 ATR) al alza con vela alcista confirming.
    """
    nombre             = "HUECO_PRO_ALC"
    direccion          = "LONG"
    MAX_DIST_SMA20_PCT = 0.0 # Excepción Barrio Sésamo

    def detectar(self, df: pd.DataFrame) -> Señal:
        n = len(df)
        idx = n - 1
        if n < 2: return SEÑAL_VACIA
        
        actual = df.iloc[idx]
        previa = df.iloc[idx-1]
        atr = float(actual.get("atr14", 0))
        
        gap = float(actual["open"]) - float(previa["high"])
        if atr > 0 and gap > 0.5 * atr and float(actual["close"]) > float(actual["open"]):
            return Señal(
                detectado=True,
                precio_entrada=round(float(actual["high"]), 4),
                idx_vela=idx,
                razon=f"Hueco PRO Alcista: Gap de {gap:.4f} (>0.5 ATR)",
                fase_mercado="Exhalación"
            )
        return SEÑAL_VACIA

class HuecoProBajista(PatronBase):
    """
    Hueco PRO Bajista (SHORT, Etapa 4).
    Gap significativo (>0.5 ATR) a la baja con vela bajista confirming.
    """
    nombre             = "HUECO_PRO_BAJ"
    direccion          = "SHORT"
    MAX_DIST_SMA20_PCT = 0.0

    def detectar(self, df: pd.DataFrame) -> Señal:
        n = len(df)
        idx = n - 1
        if n < 2: return SEÑAL_VACIA
        
        actual = df.iloc[idx]
        previa = df.iloc[idx-1]
        atr = float(actual.get("atr14", 0))
        
        gap = float(previa["low"]) - float(actual["open"])
        if atr > 0 and gap > 0.5 * atr and float(actual["close"]) < float(actual["open"]):
            return Señal(
                detectado=True,
                precio_entrada=round(float(actual["low"]), 4),
                idx_vela=idx,
                razon=f"Hueco PRO Bajista: Gap de {gap:.4f} (>0.5 ATR)",
                fase_mercado="Exhalación"
            )
        return SEÑAL_VACIA

class BottomTailEncubierta(PatronBase):
    """
    BT Encubierta (LONG, Etapa 2).
    Cola inferior >= 60% del rango total. Rechazo en soporte/SMA20.
    """
    nombre    = "BT_ENC"
    direccion = "LONG"

    def detectar(self, df: pd.DataFrame) -> Señal:
        n = len(df)
        idx = n - 1
        ultimo = df.iloc[idx]
        
        rango = float(ultimo["high"]) - float(ultimo["low"])
        if rango == 0: return SEÑAL_VACIA
        
        cola_inf = min(float(ultimo["open"]), float(ultimo["close"])) - float(ultimo["low"])
        if cola_inf / rango >= 0.60:
            return Señal(
                detectado=True,
                precio_entrada=round(float(ultimo["high"]), 4),
                idx_vela=idx,
                razon=f"BT Encubierta: cola inferior del {cola_inf/rango:.1%}",
                fase_mercado="Inhalación"
            )
        return SEÑAL_VACIA

class TopTailEncubierta(PatronBase):
    """
    TT Encubierta (SHORT, Etapa 4).
    Cola superior >= 60% del rango total. Rechazo en resistencia/SMA20.
    """
    nombre    = "TT_ENC"
    direccion = "SHORT"

    def detectar(self, df: pd.DataFrame) -> Señal:
        n = len(df)
        idx = n - 1
        ultimo = df.iloc[idx]
        
        rango = float(ultimo["high"]) - float(ultimo["low"])
        if rango == 0: return SEÑAL_VACIA
        
        cola_sup = float(ultimo["high"]) - max(float(ultimo["open"]), float(ultimo["close"]))
        if cola_sup / rango >= 0.60:
            return Señal(
                detectado=True,
                precio_entrada=round(float(ultimo["low"]), 4),
                idx_vela=idx,
                razon=f"TT Encubierta: cola superior del {cola_sup/rango:.1%}",
                fase_mercado="Inhalación"
            )
        return SEÑAL_VACIA
