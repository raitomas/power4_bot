"""
power4_bot/core/logging_config.py
================================================
Configura el sistema de logging con 3 niveles:
INFO, TRADE y ERROR — con rotación diaria.
================================================
"""

import logging
import logging.handlers
import os
import yaml


# Nivel personalizado TRADE (entre INFO y WARNING)
TRADE_LEVEL = 25
logging.addLevelName(TRADE_LEVEL, "TRADE")


def trade(self, message, *args, **kwargs):
    """Método helper para llamar logger.trade(...)"""
    if self.isEnabledFor(TRADE_LEVEL):
        self._log(TRADE_LEVEL, message, args, **kwargs)


logging.Logger.trade = trade


def configurar_logging(settings_path: str = None) -> None:
    """
    Inicializa el sistema de logging desde settings.yaml.
    Crea handlers para consola + archivo rotativo.
    """
    # Cargar settings
    if settings_path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        settings_path = os.path.join(base, "config", "settings.yaml")

    cfg = {}
    if os.path.exists(settings_path):
        with open(settings_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            cfg = data.get("logging", {})

    nivel_str  = cfg.get("level", "INFO")
    archivo    = cfg.get("archivo", "logs/power4.log")
    max_bytes  = cfg.get("max_bytes", 5 * 1024 * 1024)
    backups    = cfg.get("backup_count", 7)

    nivel = getattr(logging, nivel_str.upper(), logging.INFO)

    # Crear directorio de logs si no existe
    os.makedirs(os.path.dirname(archivo) if os.path.dirname(archivo) else ".", exist_ok=True)

    # Formato de los mensajes
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler 1: Consola (colorizado por nivel)
    console = logging.StreamHandler()
    console.setLevel(nivel)
    console.setFormatter(_ColorFormatter(fmt))

    # Handler 2: Archivo rotativo por tamaño
    file_handler = logging.handlers.RotatingFileHandler(
        filename=archivo,
        maxBytes=max_bytes,
        backupCount=backups,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    logging.getLogger("power4_bot").info(
        f"Sistema de logging iniciado | Nivel: {nivel_str} | Archivo: {archivo}"
    )


class _ColorFormatter(logging.Formatter):
    """Añade colores ANSI a la salida de consola por nivel."""

    COLORES = {
        "DEBUG":   "\033[37m",   # gris
        "INFO":    "\033[36m",   # cyan
        "TRADE":   "\033[32m",   # verde
        "WARNING": "\033[33m",   # amarillo
        "ERROR":   "\033[31m",   # rojo
        "CRITICAL":"\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def __init__(self, formatter: logging.Formatter):
        super().__init__()
        self._base = formatter

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORES.get(record.levelname, "")
        msg = self._base.format(record)
        return f"{color}{msg}{self.RESET}"
