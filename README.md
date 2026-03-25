# Power 4 Trading Bot 🤖💹

**Power 4 Bot** es un sistema de trading automatizado diseñado para MetaTrader 5 (MT5), basado en la estrategia integral del **Método Power 4**. El bot sincroniza múltiples marcos temporales (Semanal y Diario) para identificar activos con alta probabilidad de tendencia y ejecuta operaciones basadas en patrones de velas confirmados.

## 🚀 Características Principales

-   **Análisis Multi-Frame (W1 + D1)**: Cumplimiento estricto de la "Regla de Oro".
-   **Clasificador de Etapas**: Identificación automática de Acumulación, Alcista, Distribución y Bajista.
-   **Scanner de Patrones**: Detección de PC1, PV1, Fallos, Búsquedas de Liquidez y más.
-   **Gestión de Riesgo Integrada**: Cálculo automático de Stop Loss (regla del 1%), Take Profit y tamaño de posición basado en el capital.
-   **Dashboard Interactivo**: Visualización en tiempo real con Streamlit.
-   **Modos de Ejecución**: Soporta modo *Paper Trading* (simulado) y *Live Trading*.

## 📂 Estructura del Proyecto

```text
power4_bot/
├── config/             # Configuración (Watchlist, Settings)
├── core/               # Conector MT5, Data Fetcher, Símbolos
├── engine/             # Lógica de la estrategia (Indicadores, Etapas, Patrones)
├── execution/          # Gestión de órdenes y Risk Manager
├── dashboard/          # Interfaz de usuario (Streamlit)
├── backtesting/        # Motor de pruebas históricas
├── data/               # Caché de datos OHLC (no incluido en git)
└── logs/               # Registros de actividad (no incluido en git)
```

## 🛠️ Instalación

1.  **Requisitos**: Windows (necesario para la API de MetaTrader 5) y Python 3.10+.
2.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/tu-usuario/power4_bot.git
    cd power4_bot
    ```
3.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configuración**:
    -   Modifica `config/watchlist.yaml` para añadir tus activos.
    -   Configura tus parámetros de riesgo en `config/settings.yaml`.
    -   Asegúrate de que MetaTrader 5 esté abierto y logueado en tu cuenta.

## 🏃 Cómo Ejecutar

### Modo Dashboard (Recomendado)
Para arrancar la interfaz visual y el bot en segundo plano:
```bash
streamlit run dashboard/app.py
```
O simplemente ejecuta el archivo `ARRANCAR_BOT.bat`.

### Modo Terminal (Verificación)
```bash
python main.py
```

## ⚠️ Descargo de Responsabilidad
Este software es para fines educativos y de investigación. El trading conlleva un riesgo significativo de pérdida de capital. El autor no se hace responsable de las pérdidas financieras derivadas del uso de este bot.

## 📄 Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
