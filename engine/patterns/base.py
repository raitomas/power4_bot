"""
power4_bot/engine/patterns/base.py
================================================
Clase base abstracta para todos los patrones.
Define la interfaz común y las precondiciones
que TODOS los patrones deben verificar antes
de ejecutar su lógica específica.
================================================
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Señal:
    """
    Resultado de un detector de patrón.
    Si detectado=False, todos los demás campos son irrelevantes.

    Modos de entrada:
      · "confirmado"   — el trigger ya ocurrió (entrada a mercado / BUY_STOP muy cerca)
      · "condicional"  — el patrón está formado pero el trigger AÚN no se dio.
                         Se coloca un BUY_STOP / SELL_STOP en `nivel_activacion`.
    """
    detectado:      bool   = False
    patron:         str    = ""         # "PC1", "PV1", "123_ALC", etc.
    symbol:         str    = ""
    direccion:      str    = ""         # "LONG" o "SHORT"
    precio_entrada: float  = 0.0        # Nivel donde colocar la orden
    stop_loss:      float  = 0.0        # Stop Loss calculado
    take_profit:    float  = 0.0        # Take Profit sugerido (0 = no calculado aún)
    ratio_rr:       float  = 0.0        # R/R estimado
    idx_vela:       int    = -1         # Índice de la vela de trigger
    razon:          str    = ""         # Explicación detallada
    fuerza:         int    = 1          # 1=normal, 2=fuerte, 3=excepcional
    fase_mercado:   str    = ""         # "Inhalación" o "Exhalación"
    # ── Campos para orden CONDICIONAL ─────────────────────────────
    modo_entrada:       str   = "confirmado"  # "confirmado" | "condicional"
    nivel_activacion:   float = 0.0           # Precio de activación (BUY/SELL STOP)
    pendiente_expira:   int   = 3             # Días antes de cancelar si no activa
    # Datos extra del patrón (velas clave, niveles, etc.)
    datos_extra:    dict   = field(default_factory=dict)

    def __bool__(self):
        return self.detectado

    def es_condicional(self) -> bool:
        return self.modo_entrada == "condicional"

    def __repr__(self):
        if not self.detectado:
            return f"Señal(❌ {self.patron})"
        tag = "🕐" if self.es_condicional() else "✅"
        return (
            f"Señal({tag} {self.patron} {self.direccion} "
            f"entrada={self.precio_entrada:.4f} "
            f"sl={self.stop_loss:.4f} "
            f"rr={self.ratio_rr:.1f}:1)"
        )


# Señal vacía reutilizable
SEÑAL_VACIA = Señal(detectado=False)


class PatronBase(ABC):
    """
    Clase base para todos los detectores de patrones.

    Flujo de ejecución:
      1. verificar_precondiciones() — común a todos
      2. detectar()                 — lógica específica del patrón
      3. calcular_stop_loss()       — regla del 1%
      4. calcular_take_profit()     — buscar fricción a la izquierda
    """

    nombre:     str = "BASE"
    direccion:  str = ""   # "LONG" o "SHORT"

    # Parámetros globales (pueden sobreescribirse por patrón)
    BUFFER_ENTRADA_PCT  = 0.001   # 0.1% sobre/bajo el trigger
    STOP_BUFFER_PCT     = 0.01    # 1% por debajo/encima del ref de stop
    MIN_DIST_SMA20_PCT  = 0.0
    MAX_DIST_SMA20_PCT  = 4.0     # Regla Barrio Sésamo
    LOOKBACK_FRICCION   = 60      # Velas hacia atrás para buscar TP

    def evaluar(
        self,
        df:     pd.DataFrame,
        symbol: str = "",
    ) -> Señal:
        """
        Método público principal. Orquesta todo el proceso.

        Args:
            df:     DataFrame OHLC enriquecido con indicadores
            symbol: Nombre del activo (para logging)

        Returns:
            Señal con todos los campos calculados, o SEÑAL_VACIA
        """
        # 1. Precondiciones comunes
        ok, motivo = self._precondiciones(df)
        if not ok:
            logger.debug(f"{symbol} [{self.nombre}] Precondición fallida: {motivo}")
            return SEÑAL_VACIA

        # 2. Lógica específica del patrón
        señal = self.detectar(df)
        if not señal.detectado:
            return señal

        señal.patron  = self.nombre
        señal.symbol  = symbol
        señal.direccion = self.direccion

        # 3. Stop Loss (si no lo calculó el detector)
        if señal.stop_loss == 0.0:
            señal.stop_loss = self.calcular_stop_loss(df, señal)

        # 4. Take Profit (si no lo calculó el detector)
        if señal.take_profit == 0.0:
            señal.take_profit = self.calcular_take_profit(df, señal)

        # 5. Ratio R/R
        if señal.stop_loss > 0 and señal.take_profit > 0:
            riesgo   = abs(señal.precio_entrada - señal.stop_loss)
            beneficio = abs(señal.take_profit - señal.precio_entrada)
            señal.ratio_rr = round(beneficio / riesgo, 2) if riesgo > 0 else 0.0

        logger.info(
            f"{symbol:>6} [{self.nombre}] {señal}"
        )
        return señal

    @abstractmethod
    def detectar(self, df: pd.DataFrame) -> Señal:
        """
        Implementa la lógica específica del patrón.
        Debe devolver una Señal con detectado=True/False.
        """
        ...

    def detectar_prepatron(self, df: pd.DataFrame) -> Optional["Señal"]:
        """
        Detecta el patrón casi completo (pre-trigger / condicional).
        Devuelve una Señal con modo_entrada="condicional" si el patrón
        está formado pero el trigger aún no se ha dado.
        Devuelve None si el patrón no está en esa situación.

        Los subclases que soporten órdenes condicionales sobreescriben este método.
        """
        return None

    def evaluar_prepatron(
        self,
        df:     pd.DataFrame,
        symbol: str = "",
    ) -> Optional["Señal"]:
        """
        Orquesta la detección de pre-patrón (condicional).
        Equivalente a evaluar() pero llama a detectar_prepatron().
        """
        ok, motivo = self._precondiciones(df)
        if not ok:
            return None

        señal = self.detectar_prepatron(df)
        if señal is None or not señal.detectado:
            return None

        señal.patron     = self.nombre
        señal.symbol     = symbol
        señal.direccion  = self.direccion
        señal.modo_entrada = "condicional"

        if señal.stop_loss == 0.0:
            señal.stop_loss = self.calcular_stop_loss(df, señal)
        if señal.take_profit == 0.0:
            señal.take_profit = self.calcular_take_profit(df, señal)
        if señal.stop_loss > 0 and señal.take_profit > 0:
            riesgo    = abs(señal.nivel_activacion - señal.stop_loss)
            beneficio = abs(señal.take_profit - señal.nivel_activacion)
            señal.ratio_rr = round(beneficio / riesgo, 2) if riesgo > 0 else 0.0

        logger.info(f"{symbol:>6} [{self.nombre}] PRE-PATRÓN {señal}")
        return señal



    def _precondiciones(self, df: pd.DataFrame) -> tuple[bool, str]:
        """
        Verificaciones comunes a todos los patrones:
        1. DataFrame tiene suficientes barras
        2. Indicadores calculados
        3. Precio cerca de SMA20 (Regla Barrio Sésamo)
        """
        if df is None or len(df) < 30:
            return False, "DataFrame insuficiente"

        if "sma20" not in df.columns:
            return False, "Indicadores no calculados (falta sma20)"

        ultimo = df.iloc[-1]

        # Regla Barrio Sésamo (excepto patrones que operan en exhalación)
        if self.MAX_DIST_SMA20_PCT > 0:
            dist = float(ultimo.get("dist_sma20_pct", 999))
            if dist > self.MAX_DIST_SMA20_PCT:
                return False, f"Precio lejos de SMA20: {dist:.1f}% > {self.MAX_DIST_SMA20_PCT}%"

        return True, ""

    def calcular_stop_loss(self, df: pd.DataFrame, señal: Señal) -> float:
        """
        Regla del 1% del Método Power 4:
        LONG:  ref = min(low_hoy, low_ayer)  → stop = ref * 0.99
        SHORT: ref = max(high_hoy, high_ayer) → stop = ref * 1.01
        """
        if len(df) < 2:
            return 0.0

        ult  = df.iloc[-1]
        prev = df.iloc[-2]

        if self.direccion == "LONG":
            ref  = min(float(ult["low"]), float(prev["low"]))
            stop = round(ref * (1 - self.STOP_BUFFER_PCT), 4)
        else:
            ref  = max(float(ult["high"]), float(prev["high"]))
            stop = round(ref * (1 + self.STOP_BUFFER_PCT), 4)

        return stop

    def calcular_take_profit(self, df: pd.DataFrame, señal: Señal) -> float:
        """
        Busca la primera zona de fricción a la izquierda:
        - Swing highs previos (para LONG)
        - Swing lows previos (para SHORT)
        Coloca el TP justo ANTES de esa zona.
        """
        ventana = df.iloc[-self.LOOKBACK_FRICCION:]

        if self.direccion == "LONG":
            # Buscar swing highs por encima del precio de entrada
            swings = ventana[
                (ventana["swing_high"] == True) &
                (ventana["swing_high_price"] > señal.precio_entrada)
            ]["swing_high_price"]

            if len(swings) == 0:
                # Sin referencia clara → usar extensión 2R del riesgo
                riesgo = abs(señal.precio_entrada - señal.stop_loss)
                return round(señal.precio_entrada + riesgo * 2, 4)

            # TP = primer swing high por encima. Asegurar que sea > entrada
            primer_swing = swings.min()
            tp = round(primer_swing * 0.998, 4)  # Margen más pequeño (0.2%)
            
            # Si el margen lo deja por debajo de la entrada, usar un R/R mínimo de 1.1
            if tp <= señal.precio_entrada:
                riesgo = abs(señal.precio_entrada - señal.stop_loss)
                tp = round(señal.precio_entrada + riesgo * 1.1, 4)
            return tp

        else:  # SHORT
            swings = ventana[
                (ventana["swing_low"] == True) &
                (ventana["swing_low_price"] < señal.precio_entrada)
            ]["swing_low_price"]

            if len(swings) == 0:
                riesgo = abs(señal.precio_entrada - señal.stop_loss)
                return round(señal.precio_entrada - riesgo * 2, 4)

            # TP = primer swing low por debajo. Asegurar que sea < entrada
            primer_swing = swings.max()
            tp = round(primer_swing * 1.002, 4)
            
            if tp >= señal.precio_entrada:
                riesgo = abs(señal.precio_entrada - señal.stop_loss)
                tp = round(señal.precio_entrada - riesgo * 1.1, 4)
            return tp
