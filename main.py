"""
power4_bot/main.py
================================================
Entry point principal del bot.
Ejecuta el ciclo completo: conexión → datos →
análisis → señales → gestión de posiciones.

USO:
    python main.py              # Ciclo completo
    python main.py --fase1      # Solo verificar conexión y datos
    python main.py --symbol AAPL # Analizar un símbolo concreto
================================================
"""

import argparse
import logging
import sys
import os

# Aseguramos que el directorio raíz esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.logging_config import configurar_logging
from core.mt5_connector import conectar, desconectar, get_account_info, esta_conectado
from core.symbols import get_simbolos_validos
from core.data_fetcher import descargar_ohlc, actualizar_incremental

logger = logging.getLogger("power4_bot.main")


def fase1_verificacion(args) -> bool:
    """
    Fase 1: Verifica conexión MT5 y descarga datos OHLC.
    Imprime un resumen de los últimos 5 registros por activo.
    """
    logger.info("=" * 60)
    logger.info("  POWER 4 BOT — FASE 1: Conexión + Datos")
    logger.info("=" * 60)

    # ── Conexión ────────────────────────────────────────────────
    logger.info("Conectando con MetaTrader 5...")
    if not conectar():
        logger.error("No se pudo conectar con MT5. Verifica config/mt5_credentials.yaml")
        return False

    cuenta = get_account_info()
    logger.info(
        f"Cuenta activa: {cuenta.get('login')} | "
        f"Balance: ${cuenta.get('balance', 0):,.2f} | "
        f"Broker: {cuenta.get('company', 'N/A')}"
    )

    # ── Carga de símbolos ────────────────────────────────────────
    simbolos = get_simbolos_validos()
    if not simbolos:
        logger.error("No hay símbolos válidos en la watchlist.")
        desconectar()
        return False

    logger.info(f"Watchlist validada: {len(simbolos)} símbolos activos")

    # Si se especificó un símbolo concreto, filtrar
    if hasattr(args, "symbol") and args.symbol:
        simbolos = [s for s in simbolos if s["symbol"] == args.symbol.upper()]
        if not simbolos:
            logger.error(f"Símbolo no encontrado en watchlist: {args.symbol}")
            desconectar()
            return False

    # ── Descarga de datos ────────────────────────────────────────
    resultados = []
    errores = []

    for activo in simbolos:
        sym = activo["symbol"]
        logger.info(f"Descargando {sym} ({activo['tipo']})...")

        # Datos diarios (300 barras para cubrir SMA200)
        df_d1 = descargar_ohlc(sym, timeframe="D1", n_barras=300)
        # Datos semanales (80 barras)
        df_w1 = descargar_ohlc(sym, timeframe="W1", n_barras=80)

        if df_d1 is None or df_w1 is None:
            errores.append(sym)
            continue

        resultados.append({
            "symbol":    sym,
            "tipo":      activo["tipo"],
            "barras_d1": len(df_d1),
            "barras_w1": len(df_w1),
            "desde_d1":  df_d1.index[0].date(),
            "hasta_d1":  df_d1.index[-1].date(),
            "close_ult": df_d1["close"].iloc[-1],
        })

    # ── Resumen ──────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 60)
    logger.info(f"  RESUMEN FASE 1 — {len(resultados)}/{len(simbolos)} OK")
    logger.info("─" * 60)
    logger.info(f"  {'SÍMBOLO':<10} {'TIPO':<15} {'BARRAS D1':>10} {'BARRAS W1':>10} {'ÚLTIMO CLOSE':>14}")
    logger.info("  " + "-" * 58)

    for r in resultados:
        logger.info(
            f"  {r['symbol']:<10} {r['tipo']:<15} "
            f"{r['barras_d1']:>10} {r['barras_w1']:>10} "
            f"${r['close_ult']:>12.4f}"
        )

    if errores:
        logger.warning(f"\n  Símbolos con error: {', '.join(errores)}")

    logger.info("─" * 60)
    logger.info("Fase 1 completada ✓")

    # ── Muestra las últimas 5 velas del primer símbolo ───────────
    if resultados:
        sym_demo = resultados[0]["symbol"]
        df_demo = descargar_ohlc(sym_demo, "D1")
        if df_demo is not None:
            logger.info(f"\n  Últimas 5 velas diarias de {sym_demo}:")
            logger.info(f"  {'FECHA':<12} {'OPEN':>10} {'HIGH':>10} {'LOW':>10} {'CLOSE':>10}")
            for idx, row in df_demo.tail(5).iterrows():
                alcista = "▲" if row["close"] >= row["open"] else "▼"
                logger.info(
                    f"  {str(idx.date()):<12} "
                    f"{row['open']:>10.4f} {row['high']:>10.4f} "
                    f"{row['low']:>10.4f} {row['close']:>10.4f} {alcista}"
                )

    desconectar()
    return True


def main():
    parser = argparse.ArgumentParser(description="Power 4 Trading Bot")
    parser.add_argument("--fase1", action="store_true", help="Ejecutar solo Fase 1 (conexión + datos)")
    parser.add_argument("--symbol", type=str, help="Analizar un símbolo concreto")
    parser.add_argument("--actualizar", action="store_true", help="Actualización incremental de caché")
    args = parser.parse_args()

    # Inicializar logging
    configurar_logging()

    if args.fase1 or True:  # Por defecto ejecuta Fase 1 hasta que las demás estén listas
        exito = fase1_verificacion(args)
        sys.exit(0 if exito else 1)


if __name__ == "__main__":
    main()
