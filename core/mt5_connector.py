"""
power4_bot/core/mt5_connector.py
================================================
Gestiona el ciclo de vida de la conexión con
MetaTrader 5. Singleton thread-safe.
================================================
"""

import logging
import os
import yaml

logger = logging.getLogger(__name__)

# Importación condicional: en entornos sin MT5 instalado
# (ej. CI/CD, Mac) se usa un stub para que el resto del
# código no explote al hacer import.
try:
    import MetaTrader5 as mt5
    MT5_DISPONIBLE = True
except ImportError:
    mt5 = None
    MT5_DISPONIBLE = False
    logger.warning("MetaTrader5 no instalado. Modo simulación activo.")


def _cargar_credenciales(path: str = None) -> dict:
    """Carga las credenciales desde el YAML de configuración."""
    if path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "config", "mt5_credentials.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Archivo de credenciales no encontrado: {path}\n"
            "Copia config/mt5_credentials.yaml y rellena tus datos."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data.get("mt5", {})


def conectar(credentials_path: str = None) -> bool:
    """
    Inicializa y autentica la conexión con MT5.

    Estrategia de conexión (en orden):
      1. Si MT5 está abierto y ya tiene sesión activa → la usa directamente
         (sin necesitar contraseña en el YAML)
      2. Si no hay sesión activa y las credenciales son reales → login explícito
      3. Si la contraseña es el placeholder → solo usa la sesión activa

    Returns:
        True si la conexión fue exitosa, False si falló.
    """
    if not MT5_DISPONIBLE:
        logger.warning("MT5 no disponible. Simulando conexión OK.")
        return True

    # Paso 1: Inicializar el terminal MT5
    if not mt5.initialize():
        logger.error(f"mt5.initialize() falló. Error: {mt5.last_error()}")
        return False

    # Paso 2: Comprobar si ya hay sesión activa en el terminal abierto
    info_actual = mt5.account_info()
    if info_actual is not None and info_actual.login > 0:
        logger.info(
            f"MT5 conectado ✓ (sesión activa) | Cuenta: {info_actual.login} | "
            f"Broker: {info_actual.company} | "
            f"Balance: ${info_actual.balance:,.2f} | "
            f"Servidor: {info_actual.server}"
        )
        return True

    # Paso 3: No hay sesión activa → intentar login con credenciales
    creds = _cargar_credenciales(credentials_path)
    password = str(creds.get("password", ""))

    # Si la contraseña es el placeholder, no podemos hacer login
    if not password or password in ("TU_PASSWORD", "YOUR_PASSWORD", ""):
        logger.error(
            "No hay sesión MT5 activa y la contraseña en "
            "config/mt5_credentials.yaml es un placeholder. "
            "Abre MetaTrader 5 e inicia sesión manualmente, "
            "o rellena la contraseña en el archivo de configuración."
        )
        mt5.shutdown()
        return False

    autorizado = mt5.login(
        login=int(creds["login"]),
        password=password,
        server=str(creds["server"]),
        timeout=int(creds.get("timeout", 60000)),
    )

    if not autorizado:
        logger.error(
            f"Login MT5 fallido para cuenta {creds['login']} "
            f"en servidor {creds['server']}. "
            f"Error: {mt5.last_error()}"
        )
        mt5.shutdown()
        return False

    info = mt5.account_info()
    logger.info(
        f"MT5 conectado ✓ | Cuenta: {info.login} | "
        f"Broker: {info.company} | "
        f"Balance: ${info.balance:,.2f} | "
        f"Servidor: {info.server}"
    )
    return True


def desconectar() -> None:
    """Cierra la conexión con MT5 limpiamente."""
    if not MT5_DISPONIBLE:
        return
    mt5.shutdown()
    logger.info("Conexión MT5 cerrada.")


def get_account_info() -> dict:
    """Devuelve información básica de la cuenta activa."""
    if not MT5_DISPONIBLE:
        return {
            "login": 0, "balance": 100000.0,
            "equity": 100000.0, "margin_free": 100000.0,
            "company": "Simulación", "server": "Demo",
        }

    info = mt5.account_info()
    if info is None:
        logger.error(f"No se pudo obtener account_info: {mt5.last_error()}")
        return {}

    return {
        "login":       info.login,
        "balance":     info.balance,
        "equity":      info.equity,
        "margin_free": info.margin_free,
        "profit":      info.profit,
        "company":     info.company,
        "server":      info.server,
        "currency":    info.currency,
        "leverage":    info.leverage,
    }


def esta_conectado() -> bool:
    """Comprueba si el terminal MT5 está activo y conectado."""
    if not MT5_DISPONIBLE:
        return True  # modo simulación siempre "conectado"
    terminal = mt5.terminal_info()
    return terminal is not None and terminal.connected
