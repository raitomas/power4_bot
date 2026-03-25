"""
check_symbols.py
================================================
Pequeño script de utilidad para verificar qué símbolos
de la watchlist están disponibles en tu broker actual.
"""
import os
import sys

# Añadir raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from core.mt5_connector import conectar, desconectar
from core.symbols import _cargar_watchlist, validar_simbolo

def main():
    print("=" * 60)
    print("  VERIFICADOR DE SÍMBOLOS - POWER 4 BOT")
    print("=" * 60)

    if not conectar():
        print("❌ Error: No se pudo conectar a MetaTrader 5.")
        return

    print("Conectado a MT5 (OK)\n")
    
    todos = _cargar_watchlist()
    disponibles = []
    faltantes = []

    for activo in todos:
        sym = activo["symbol"]
        if validar_simbolo(sym):
            disponibles.append(sym)
            print(f"[OK] {sym:<10} - Disponible")
        else:
            faltantes.append(sym)
            print(f"[X]  {sym:<10} - NO ENCONTRADO (Revisar sufijos)")

    print("\n" + "=" * 60)
    print(f"RESUMEN: {len(disponibles)} disponibles / {len(faltantes)} faltantes.")
    print("=" * 60)
    
    print("\nBuscando coincidencias para los símbolos faltantes en TODO el broker...")
    print("\nBúsqueda PROFUNDA de Índices y Petróleo...")
    all_symbols = mt5.symbols_get()
    if all_symbols:
        keywords = ["100", "50", "35", "40", "200", "30", "OIL", "WTI", "BRENT"]
        print("\nSímbolos que coinciden con patrones de Índices/Energía:")
        for kw in keywords:
            found = [s.name for s in all_symbols if kw in s.name.upper()]
            if found:
                print(f"  - {kw}: {', '.join(found[:10])}")

    desconectar()

if __name__ == "__main__":
    main()
