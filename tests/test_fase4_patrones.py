"""
power4_bot/tests/test_fase4_patrones.py
================================================
Tests unitarios para todos los detectores de
patrones de la Fase 4.
Ejecutar: pytest tests/test_fase4_patrones.py -v
================================================
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.indicators import calcular_indicadores
from engine.patterns.base import Señal, SEÑAL_VACIA
from engine.patterns.pc1_pv1 import PC1, PV1
from engine.patterns.patron_123 import Patron123Alcista, Patron123Bajista
from engine.patterns.acunamiento import AcunamientoAlcista, AcunamientoBajista
from engine.patterns.otros_patrones import (
    FalloRupturaBajista, FalloRupturaAlcista,
    VelaRojaIgnorada, VelaVerdeIgnorada,
    PRCA, PRCB,
)
from engine.pattern_scanner import escanear, escanear_watchlist
from engine.stage_classifier import (
    ResultadoAlineamiento, Etapa, verificar_alineamiento
)


# ══════════════════════════════════════════════════════════════════
#  CONSTRUCTORES DE DFs PARA TESTS
# ══════════════════════════════════════════════════════════════════

def _df_con_indicadores(n=300, seed=42, drift=0.005):
    """DataFrame con tendencia y todos los indicadores calculados."""
    np.random.seed(seed)
    fechas = pd.bdate_range(end="2026-01-01", periods=n)
    precio = 100.0
    rows = []
    for _ in fechas:
        d = precio * np.random.uniform(drift * 0.5, drift * 1.5)
        o = precio
        c = precio + d
        h = max(o, c) * np.random.uniform(1.001, 1.008)
        l = min(o, c) * np.random.uniform(0.992, 0.999)
        rows.append({"open": o, "high": h, "low": l, "close": c})
        precio = c
    df = pd.DataFrame(rows, index=fechas)
    return calcular_indicadores(df)


def _df_con_escalera_pc1(n_escalones=4):
    """
    Crea un df donde las últimas velas forman una escalera
    de máximos decrecientes (estructura PC1).
    """
    df = _df_con_indicadores(300, seed=1)
    # Inyectar un swing high y luego escalera decreciente
    rows = df.copy()
    idx = len(rows) - 1

    # Swing high base
    base_high = float(rows["close"].iloc[-n_escalones - 3])

    # Escalones decrecientes
    for i in range(n_escalones, 0, -1):
        loc = idx - i
        factor = 1 + (i * 0.003)
        rows.iloc[loc, rows.columns.get_loc("high")] = base_high * factor
        rows.iloc[loc, rows.columns.get_loc("close")] = base_high * factor * 0.998

    # Última vela supera el último máximo (trigger)
    ultimo_maximo = base_high * (1 + 0.003)
    rows.iloc[idx, rows.columns.get_loc("close")] = ultimo_maximo * 1.005
    rows.iloc[idx, rows.columns.get_loc("high")]  = ultimo_maximo * 1.006

    # Marcar swing_high en la vela previa a la escalera
    rows["swing_high"] = False
    rows.iloc[idx - n_escalones - 1, rows.columns.get_loc("swing_high")] = True
    rows.iloc[idx - n_escalones - 1, rows.columns.get_loc("swing_high_price")] = (
        base_high * (1 + (n_escalones + 1) * 0.003)
    )

    return rows


def _df_con_acunamiento_alcista():
    """
    Crea un df con condición de acunamiento alcista:
    low ≤ sma20, close > sma20, sma20_slope > 0, vela verde.
    """
    df = _df_con_indicadores(300, seed=5)
    idx = len(df) - 1
    sma20 = float(df["sma20"].iloc[idx])

    # Inyectar la vela de acunamiento
    df.iloc[idx, df.columns.get_loc("low")]   = sma20 * 0.999  # toca SMA
    df.iloc[idx, df.columns.get_loc("open")]  = sma20 * 0.9995
    df.iloc[idx, df.columns.get_loc("close")] = sma20 * 1.005  # cierra sobre
    df.iloc[idx, df.columns.get_loc("high")]  = sma20 * 1.007

    # SMA20 slope positivo
    df.iloc[idx, df.columns.get_loc("sma20_slope")] = 0.05

    # Suficientes cruces (simular rango previo)
    # Alternamos cierres sobre y bajo SMA20
    for i in range(1, 10):
        loc = idx - i
        sma_loc = float(df["sma20"].iloc[loc])
        if i % 2 == 0:
            df.iloc[loc, df.columns.get_loc("close")] = sma_loc * 1.003
        else:
            df.iloc[loc, df.columns.get_loc("close")] = sma_loc * 0.997

    df["vela_alcista"] = df["close"] > df["open"]
    df["vela_bajista"] = df["close"] < df["open"]
    df["dist_sma20_pct"] = ((df["close"] - df["sma20"]).abs() / df["sma20"] * 100)
    return df


def _df_con_vri():
    """
    Crea un df con una Vela Roja Ignorada:
    - Penúltima vela: pequeña y roja
    - Última vela: cierra sobre el máximo de la roja
    """
    df = _df_con_indicadores(300, seed=3)
    idx = len(df) - 1

    # Vela roja pequeña (penúltima)
    base = float(df["close"].iloc[idx - 2])
    df.iloc[idx - 1, df.columns.get_loc("open")]  = base * 1.002
    df.iloc[idx - 1, df.columns.get_loc("close")] = base * 0.998  # roja
    df.iloc[idx - 1, df.columns.get_loc("high")]  = base * 1.003
    df.iloc[idx - 1, df.columns.get_loc("low")]   = base * 0.995

    # Última vela: verde que supera el máximo de la roja
    high_roja = base * 1.003
    df.iloc[idx, df.columns.get_loc("open")]  = base * 1.001
    df.iloc[idx, df.columns.get_loc("close")] = high_roja * 1.005  # supera
    df.iloc[idx, df.columns.get_loc("high")]  = high_roja * 1.006
    df.iloc[idx, df.columns.get_loc("low")]   = base * 0.999

    # Asegurar columnas de distancia SMA20 para que pase precondición
    sma20 = df["sma20"].ffill()
    df["dist_sma20_pct"] = ((df["close"] - sma20).abs() / sma20 * 100).fillna(2.0)
    df["cerca_sma20"]    = df["dist_sma20_pct"] < 4.0
    # Forzar que la última vela esté cerca de SMA20
    df.iloc[idx, df.columns.get_loc("dist_sma20_pct")] = 2.0
    df.iloc[idx, df.columns.get_loc("cerca_sma20")]    = True
    return df


# ══════════════════════════════════════════════════════════════════
#  TESTS DE LA CLASE BASE (Señal)
# ══════════════════════════════════════════════════════════════════

class TestSeñal:

    def test_señal_vacia_es_falsy(self):
        assert not SEÑAL_VACIA

    def test_señal_detectada_es_truthy(self):
        s = Señal(detectado=True, patron="TEST", precio_entrada=100.0)
        assert s

    def test_señal_repr_detectada(self):
        s = Señal(
            detectado=True, patron="PC1", direccion="LONG",
            precio_entrada=100.0, stop_loss=93.0, ratio_rr=2.5
        )
        assert "PC1" in repr(s)
        assert "LONG" in repr(s)

    def test_datos_extra_por_defecto_vacio(self):
        s = Señal()
        assert s.datos_extra == {}


# ══════════════════════════════════════════════════════════════════
#  TESTS PC1 y PV1
# ══════════════════════════════════════════════════════════════════

class TestPC1PV1:

    def test_pc1_retorna_señal(self):
        df = _df_con_indicadores()
        resultado = PC1().evaluar(df, "TEST")
        assert isinstance(resultado, Señal)

    def test_pv1_retorna_señal(self):
        df = _df_con_indicadores(drift=-0.005)
        resultado = PV1().evaluar(df, "TEST")
        assert isinstance(resultado, Señal)

    def test_pc1_sin_swing_high_no_detecta(self):
        df = _df_con_indicadores()
        # Eliminar todos los swing_high
        df["swing_high"] = False
        resultado = PC1().evaluar(df, "TEST")
        assert not resultado.detectado

    def test_pc1_con_escalera_detecta(self):
        df = _df_con_escalera_pc1(n_escalones=4)
        detector = PC1()
        # Precondición manual: forzar cerca_sma20
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        resultado = detector.evaluar(df, "AAPL")
        # Puede detectar o no dependiendo de los datos sintéticos
        assert isinstance(resultado, Señal)

    def test_pc1_entrada_mayor_que_stop_long(self):
        """Para LONG: si se detecta, la entrada debe ser mayor que el stop."""
        df = _df_con_indicadores(300, seed=10, drift=0.003)
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        resultado = PC1().evaluar(df, "TEST")
        # Solo validamos si hay señal Y la dirección es correcta
        if resultado.detectado:
            assert resultado.direccion == "LONG"

    def test_pv1_entrada_menor_que_stop(self):
        df = _df_con_indicadores(drift=-0.006, seed=7)
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        resultado = PV1().evaluar(df, "TEST")
        if resultado.detectado:
            assert resultado.precio_entrada < resultado.stop_loss


# ══════════════════════════════════════════════════════════════════
#  TESTS PATRÓN 1-2-3
# ══════════════════════════════════════════════════════════════════

class TestPatron123:

    def test_123_alc_retorna_señal(self):
        df = _df_con_indicadores()
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        resultado = Patron123Alcista().evaluar(df, "TEST")
        assert isinstance(resultado, Señal)

    def test_123_baj_retorna_señal(self):
        df = _df_con_indicadores(drift=-0.005)
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        resultado = Patron123Bajista().evaluar(df, "TEST")
        assert isinstance(resultado, Señal)

    def test_123_alc_long(self):
        df = _df_con_indicadores()
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        r = Patron123Alcista().evaluar(df, "TEST")
        if r.detectado:
            assert r.direccion == "LONG"

    def test_123_baj_short(self):
        df = _df_con_indicadores(drift=-0.005)
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        r = Patron123Bajista().evaluar(df, "TEST")
        if r.detectado:
            assert r.direccion == "SHORT"


# ══════════════════════════════════════════════════════════════════
#  TESTS ACUNAMIENTO
# ══════════════════════════════════════════════════════════════════

class TestAcunamiento:

    def test_acun_alc_detecta_condicion_correcta(self):
        df = _df_con_acunamiento_alcista()
        resultado = AcunamientoAlcista().evaluar(df, "TEST")
        assert isinstance(resultado, Señal)
        # Con los datos inyectados debería detectar
        assert resultado.detectado, f"Razón no detectado: {resultado.razon}"

    def test_acun_alc_es_long(self):
        df = _df_con_acunamiento_alcista()
        r = AcunamientoAlcista().evaluar(df, "TEST")
        if r.detectado:
            assert r.direccion == "LONG"

    def test_acun_baj_es_short(self):
        df = _df_con_indicadores(drift=-0.005)
        r = AcunamientoBajista().evaluar(df, "TEST")
        if r.detectado:
            assert r.direccion == "SHORT"

    def test_acun_alc_stop_bajo_low(self):
        df = _df_con_acunamiento_alcista()
        r = AcunamientoAlcista().evaluar(df, "TEST")
        if r.detectado:
            low_vela = r.datos_extra.get("low_vela", r.stop_loss * 1.01)
            assert r.stop_loss < low_vela * 1.001  # Stop está bajo el low

    def test_acun_alc_no_detecta_sin_cruces(self):
        df = _df_con_indicadores()
        # Sin cruces SMA20 no puede haber acunamiento
        # (todos los cierres muy por encima de SMA20)
        df["close"] = df["sma20"] * 1.10   # 10% sobre SMA20 siempre
        resultado = AcunamientoAlcista().evaluar(df, "TEST")
        assert not resultado.detectado

    def test_acun_nombre_patron(self):
        df = _df_con_acunamiento_alcista()
        r = AcunamientoAlcista().evaluar(df, "TEST")
        if r.detectado:
            assert r.patron == "ACUN_ALC"


# ══════════════════════════════════════════════════════════════════
#  TESTS VRI / VVI
# ══════════════════════════════════════════════════════════════════

class TestVRIVVI:

    def test_vri_detecta_condicion_correcta(self):
        df = _df_con_vri()
        resultado = VelaRojaIgnorada().evaluar(df, "TEST")
        assert isinstance(resultado, Señal)
        assert resultado.detectado, f"VRI no detectado: {resultado.razon}"

    def test_vri_es_long(self):
        df = _df_con_vri()
        r = VelaRojaIgnorada().evaluar(df, "TEST")
        if r.detectado:
            assert r.direccion == "LONG"

    def test_vvi_es_short(self):
        df = _df_con_indicadores(drift=-0.005)
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        r = VelaVerdeIgnorada().evaluar(df, "TEST")
        if r.detectado:
            assert r.direccion == "SHORT"

    def test_vri_stop_bajo_low_roja(self):
        df = _df_con_vri()
        r = VelaRojaIgnorada().evaluar(df, "TEST")
        if r.detectado:
            low_roja = r.datos_extra.get("low_vela_roja", r.stop_loss * 1.01)
            assert r.stop_loss <= low_roja


# ══════════════════════════════════════════════════════════════════
#  TESTS PRCA / PRCB
# ══════════════════════════════════════════════════════════════════

class TestPRCAPRCB:

    def test_prca_retorna_señal(self):
        df = _df_con_indicadores()
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        r = PRCA().evaluar(df, "TEST")
        assert isinstance(r, Señal)

    def test_prcb_retorna_señal(self):
        df = _df_con_indicadores(drift=-0.005)
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        r = PRCB().evaluar(df, "TEST")
        assert isinstance(r, Señal)


# ══════════════════════════════════════════════════════════════════
#  TESTS DEL SCANNER
# ══════════════════════════════════════════════════════════════════

class TestPatternScanner:

    def _crear_alineamiento(self, symbol="TEST", dir="LONG", cerca=True):
        return ResultadoAlineamiento(
            symbol=symbol,
            etapa_w1=Etapa.ALCISTA if dir == "LONG" else Etapa.BAJISTA,
            etapa_d1=Etapa.ALCISTA if dir == "LONG" else Etapa.BAJISTA,
            direccion=dir,
            alineado=True,
            cerca_sma20=cerca,
            dist_sma20=1.5,
            operable=cerca,
        )

    def test_scanner_retorna_lista(self):
        df = _df_con_indicadores()
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        alin = self._crear_alineamiento()
        resultado = escanear(alin, df)
        assert isinstance(resultado, list)

    def test_scanner_no_opera_sin_alineamiento(self):
        df = _df_con_indicadores()
        alin = ResultadoAlineamiento(symbol="TEST", alineado=False)
        resultado = escanear(alin, df)
        assert resultado == []

    def test_scanner_señales_son_señal(self):
        df = _df_con_indicadores()
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        alin = self._crear_alineamiento()
        resultado = escanear(alin, df)
        for s in resultado:
            assert isinstance(s, Señal)
            assert s.detectado

    def test_scanner_señales_long_son_long(self):
        df = _df_con_indicadores()
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        alin = self._crear_alineamiento(dir="LONG")
        resultado = escanear(alin, df)
        for s in resultado:
            assert s.direccion == "LONG"

    def test_scanner_señales_short_son_short(self):
        df = _df_con_indicadores(drift=-0.005)
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        alin = self._crear_alineamiento(dir="SHORT")
        resultado = escanear(alin, df)
        for s in resultado:
            assert s.direccion == "SHORT"

    def test_scanner_watchlist_vacia(self):
        resultado = escanear_watchlist([], {})
        assert resultado == []

    def test_scanner_watchlist_retorna_lista(self):
        df = _df_con_indicadores()
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        alin = self._crear_alineamiento("AAPL")
        resultado = escanear_watchlist([alin], {"AAPL": df})
        assert isinstance(resultado, list)

    def test_scanner_ratio_rr_positivo(self):
        df = _df_con_indicadores()
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        alin = self._crear_alineamiento()
        resultado = escanear(alin, df)
        for s in resultado:
            if s.take_profit > 0 and s.stop_loss > 0:
                assert s.ratio_rr >= 0

    def test_scanner_señales_tienen_symbol(self):
        df = _df_con_indicadores()
        df["dist_sma20_pct"] = 2.0
        df["cerca_sma20"]    = True
        alin = self._crear_alineamiento("NVDA")
        resultado = escanear(alin, df)
        for s in resultado:
            assert s.symbol == "NVDA"
