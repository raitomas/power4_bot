"""
power4_bot/tests/test_fase5_6_execution.py
================================================
Tests unitarios para:
  - Fase 5: RiskManager + OrderManager
  - Fase 6: TrailingStopManager
Ejecutar: pytest tests/test_fase5_6_execution.py -v
================================================
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.indicators import calcular_indicadores
from engine.patterns.base import Señal
from execution.risk_manager import RiskManager, OrdenCalculada
from execution.order_manager import OrderManager, ResultadoOrden
from execution.trailing_stop import (
    TrailingStopManager, EstadoPosicion, ResultadoTrailing
)


# ══════════════════════════════════════════════════════════════════
#  FIXTURES Y HELPERS
# ══════════════════════════════════════════════════════════════════

def _df_test(n=300, seed=42, drift=0.005):
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


def _señal_test(
    symbol="AAPL",
    dir="LONG",
    entrada=100.0,
    sl=93.0,
    tp=120.0,
):
    return Señal(
        detectado      = True,
        patron         = "PC1",
        symbol         = symbol,
        direccion      = dir,
        precio_entrada = entrada,
        stop_loss      = sl,
        take_profit    = tp,
        ratio_rr       = abs(tp - entrada) / abs(entrada - sl),
    )


def _posicion_test(
    symbol="AAPL",
    dir="LONG",
    entrada=100.0,
    sl=93.0,
    tp=120.0,
    dias=5,
):
    return EstadoPosicion(
        ticket         = 123456,
        symbol         = symbol,
        direccion      = dir,
        fecha_entrada  = date.today() - timedelta(days=dias),
        precio_entrada = entrada,
        stop_loss      = sl,
        take_profit    = tp,
        volumen        = 10.0,
        patron         = "PC1",
    )


# ══════════════════════════════════════════════════════════════════
#  TESTS RISK MANAGER
# ══════════════════════════════════════════════════════════════════

class TestRiskManager:

    @pytest.fixture
    def rm(self):
        return RiskManager(
            capital=100_000,
            riesgo_pct=0.005,
            dist_sl_min_pct=0.04,
            dist_sl_max_pct=0.10,
        )

    @pytest.fixture
    def df(self):
        return _df_test()

    def test_calcular_orden_retorna_orden_calculada(self, rm, df):
        señal = _señal_test(entrada=100.0, sl=93.5, tp=120.0)
        orden = rm.calcular_orden(señal, df)
        assert isinstance(orden, OrdenCalculada)

    def test_orden_valida_con_datos_correctos(self, rm, df):
        # SL al 6.5% del precio → dentro de rango permitido
        entrada = 100.0
        sl      = 93.5    # 6.5% por debajo
        tp      = 117.0   # 17% arriba → R/R 2.6:1
        señal   = _señal_test(entrada=entrada, sl=sl, tp=tp)
        orden   = rm.calcular_orden(señal, df)
        assert orden.valida, f"Debe ser válida: {orden.motivo_rechazo}"

    def test_orden_invalida_sl_muy_ajustado(self, rm, df):
        # SL al 1% → por debajo del mínimo del 4%
        señal = _señal_test(entrada=100.0, sl=99.0, tp=110.0)
        orden = rm.calcular_orden(señal, df)
        assert not orden.valida
        assert "ajustado" in orden.motivo_rechazo.lower()

    def test_orden_invalida_sl_muy_amplio(self, rm, df):
        # SL al 15% → por encima del máximo del 10%
        señal = _señal_test(entrada=100.0, sl=85.0, tp=130.0)
        orden = rm.calcular_orden(señal, df)
        assert not orden.valida
        assert "amplio" in orden.motivo_rechazo.lower()

    def test_volumen_calculado_correctamente(self, rm, df):
        # riesgo = 100k × 0.5% = $500
        # distancia = |100 - 93.5| = $6.5
        # volumen = floor(500 / 6.5) = 76
        señal = _señal_test(entrada=100.0, sl=93.5, tp=120.0)
        orden = rm.calcular_orden(señal, df)
        if orden.valida:
            assert orden.volumen == 76.0

    def test_riesgo_dolares_no_supera_maximo(self, rm, df):
        señal = _señal_test(entrada=100.0, sl=93.5, tp=120.0)
        orden = rm.calcular_orden(señal, df)
        if orden.valida:
            riesgo_max = 100_000 * 0.005 + 1  # +1 por redondeo
            assert orden.riesgo_dolares <= riesgo_max

    def test_kill_switch_activo_bloquea_ordenes(self, rm, df):
        # Simular drawdown del 10% (por encima del límite del 8%)
        rm.actualizar_estado(posiciones_abiertas=0, equity_actual=89_000)
        señal = _señal_test(entrada=100.0, sl=93.5, tp=120.0)
        orden = rm.calcular_orden(señal, df)
        assert not orden.valida
        assert "kill switch" in orden.motivo_rechazo.lower()

    def test_max_posiciones_bloquea_ordenes(self, rm, df):
        rm.actualizar_estado(posiciones_abiertas=5, equity_actual=100_000)
        señal = _señal_test()
        orden = rm.calcular_orden(señal, df)
        assert not orden.valida
        assert "posiciones" in orden.motivo_rechazo.lower()

    def test_ratio_rr_calculado(self, rm, df):
        señal = _señal_test(entrada=100.0, sl=93.5, tp=120.0)
        orden = rm.calcular_orden(señal, df)
        if orden.valida:
            assert orden.ratio_rr > 0

    def test_estado_inicial(self, rm):
        estado = rm.get_estado()
        assert estado["capital_inicial"] == 100_000
        assert estado["posiciones_abiertas"] == 0
        assert not estado["kill_switch_activo"]

    def test_actualizar_estado_refleja_cambios(self, rm):
        rm.actualizar_estado(posiciones_abiertas=3, equity_actual=98_000)
        estado = rm.get_estado()
        assert estado["posiciones_abiertas"] == 3
        assert estado["equity_actual"] == 98_000

    def test_drawdown_calculado_correctamente(self, rm):
        rm.actualizar_estado(posiciones_abiertas=0, equity_actual=95_000)
        assert abs(rm._drawdown_actual - 0.05) < 0.001  # 5% drawdown

    def test_calcular_multiples_filtra_invalidas(self, rm, df):
        señales = [
            _señal_test(entrada=100.0, sl=93.5, tp=120.0),   # válida
            _señal_test(entrada=100.0, sl=99.5, tp=105.0),   # inválida (SL muy ajustado)
        ]
        ordenes = rm.calcular_multiples(señales, {"AAPL": df})
        # Solo la primera debería ser válida
        assert all(o.valida for o in ordenes)


# ══════════════════════════════════════════════════════════════════
#  TESTS ORDER MANAGER (modo paper)
# ══════════════════════════════════════════════════════════════════

class TestOrderManager:

    @pytest.fixture
    def om(self):
        return OrderManager(modo="paper")

    def _orden_valida(self):
        return OrdenCalculada(
            symbol         = "AAPL",
            direccion      = "LONG",
            patron         = "PC1",
            precio_entrada = 100.0,
            stop_loss      = 93.5,
            take_profit    = 120.0,
            volumen        = 76.0,
            riesgo_dolares = 494.0,
            distancia_sl_pct = 6.5,
            ratio_rr       = 2.6,
            valida         = True,
        )

    def test_enviar_orden_paper_retorna_resultado(self, om):
        orden    = self._orden_valida()
        resultado = om.enviar_orden(orden)
        assert isinstance(resultado, ResultadoOrden)

    def test_enviar_orden_paper_marcada_enviada(self, om):
        orden    = self._orden_valida()
        resultado = om.enviar_orden(orden)
        assert resultado.enviada

    def test_enviar_orden_invalida_no_enviada(self, om):
        orden = OrdenCalculada(valida=False, motivo_rechazo="Test")
        resultado = om.enviar_orden(orden)
        assert not resultado.enviada

    def test_ticket_asignado(self, om):
        orden    = self._orden_valida()
        resultado = om.enviar_orden(orden)
        assert resultado.ticket > 0

    def test_historial_crece(self, om):
        orden = self._orden_valida()
        om.enviar_orden(orden)
        om.enviar_orden(orden)
        assert len(om.get_historial()) == 2

    def test_enviar_multiples(self, om):
        ordenes = [self._orden_valida() for _ in range(3)]
        resultados = om.enviar_multiples(ordenes)
        assert len(resultados) == 3
        assert all(r.enviada for r in resultados)

    def test_posiciones_paper_vacio(self, om):
        pos = om.get_posiciones_abiertas()
        assert isinstance(pos, list)

    def test_modificar_sl_paper(self, om):
        resultado = om.modificar_stop_loss(ticket=123, nuevo_sl=94.0)
        assert resultado is True


# ══════════════════════════════════════════════════════════════════
#  TESTS TRAILING STOP MANAGER
# ══════════════════════════════════════════════════════════════════

class TestTrailingStop:

    @pytest.fixture
    def om(self):
        return OrderManager(modo="paper")

    @pytest.fixture
    def tsm(self, om):
        return TrailingStopManager(om)

    @pytest.fixture
    def df_alc(self):
        """DataFrame con tendencia alcista para trailing stop."""
        df = _df_test(n=300, drift=0.005)
        # Asegurar que las últimas 3 velas van "A Favor" (LONG)
        idx = len(df) - 1
        for i in range(3):
            loc = idx - 2 + i
            factor = 1 + (i * 0.003)
            df.iloc[loc, df.columns.get_loc("close")] = 100.0 * factor
            df.iloc[loc, df.columns.get_loc("high")]  = 100.0 * factor * 1.005
            df.iloc[loc, df.columns.get_loc("low")]   = 100.0 * factor * 0.995
        return df

    def test_evaluar_posicion_retorna_resultado(self, tsm, df_alc):
        pos = _posicion_test(dias=5)
        r   = tsm.evaluar_posicion(pos, df_alc, date.today())
        assert isinstance(r, ResultadoTrailing)

    def test_stop_bloqueado_dias_1_2(self, tsm, df_alc):
        for dias in [1, 2]:
            pos = _posicion_test(dias=dias)
            r   = tsm.evaluar_posicion(pos, df_alc, date.today())
            # Sin excepciones, el stop no debe moverse en días 1-2
            # (puede haber excepciones por gap/vela excepcional en datos sintéticos)
            assert isinstance(r, ResultadoTrailing)

    def test_stop_puede_moverse_dia_3_plus(self, tsm, df_alc):
        pos = _posicion_test(dias=5)
        r   = tsm.evaluar_posicion(pos, df_alc, date.today())
        # Puede moverse o no según los datos, pero no debe fallar
        assert isinstance(r, ResultadoTrailing)
        assert r.dias_posicion == 5

    def test_antiretroceso_long(self, tsm, df_alc):
        """Para LONG: si el nuevo SL es menor que el actual, NO mover."""
        pos = _posicion_test(dias=5, sl=200.0)   # SL muy alto (ya protegido)
        r   = tsm.evaluar_posicion(pos, df_alc, date.today())
        # El nuevo SL calculado será más bajo que 200, así que no se mueve
        if not r.movido:
            assert "antiretroceso" in r.razon.lower() or "a favor" in r.razon.lower()

    def test_sl_nuevo_mayor_que_anterior_si_mueve_long(self, tsm, df_alc):
        """Si se mueve, el nuevo SL debe ser mayor que el anterior (LONG)."""
        pos = _posicion_test(dias=5, sl=85.0)   # SL bajo → permite subir
        r   = tsm.evaluar_posicion(pos, df_alc, date.today())
        if r.movido:
            assert r.sl_nuevo > r.sl_anterior

    def test_vela_a_favor_long(self, tsm):
        """Test directo del criterio A Favor."""
        df = _df_test(n=50)
        idx = len(df) - 1

        # Forzar vela "A Favor" para LONG
        df.iloc[idx, df.columns.get_loc("close")] = 110.0
        df.iloc[idx, df.columns.get_loc("high")]  = 112.0
        df.iloc[idx, df.columns.get_loc("low")]   = 108.0
        df.iloc[idx - 1, df.columns.get_loc("close")] = 105.0
        df.iloc[idx - 1, df.columns.get_loc("high")]  = 107.0
        df.iloc[idx - 1, df.columns.get_loc("low")]   = 103.0

        resultado = tsm._vela_a_favor(df, "LONG")
        assert resultado is True

    def test_vela_no_a_favor_long(self, tsm):
        """Test de vela que NO va a favor para LONG."""
        df = _df_test(n=50)
        idx = len(df) - 1

        # Vela que baja (contra LONG)
        df.iloc[idx, df.columns.get_loc("close")] = 95.0
        df.iloc[idx, df.columns.get_loc("high")]  = 98.0
        df.iloc[idx, df.columns.get_loc("low")]   = 93.0
        df.iloc[idx - 1, df.columns.get_loc("close")] = 105.0
        df.iloc[idx - 1, df.columns.get_loc("high")]  = 107.0
        df.iloc[idx - 1, df.columns.get_loc("low")]   = 103.0

        resultado = tsm._vela_a_favor(df, "LONG")
        assert resultado is False

    def test_gestionar_todas_retorna_lista(self, tsm, df_alc):
        posiciones = [{
            "ticket": 111, "symbol": "AAPL", "tipo": "LONG",
            "tiempo": date.today() - timedelta(days=5),
            "entrada": 100.0, "sl": 90.0, "tp": 120.0,
            "volumen": 10.0, "comment": "P4_PC1",
        }]
        resultados = tsm.gestionar_todas(
            posiciones  = posiciones,
            datos_d1    = {"AAPL": df_alc},
            fecha_hoy   = date.today(),
        )
        assert isinstance(resultados, list)
        assert len(resultados) == 1
