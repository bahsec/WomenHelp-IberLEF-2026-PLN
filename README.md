# Clasificación de reportes de violencia de género (WomenHelp 2026)

Este repositorio contiene el código fuente para los experimentos detallados en nuestro paper para la competición IberLEF 2026.

## Estructura del Código
* **Modelos Clásicos y MrBERT-es:**
  * `Visualizacion.ipynb`: Modelo clásico.
  * `MrBert.ipynb`: MrBERT-es.
* **Llama-3 (8B) con QLoRA:**
  * `entrenamiento_llama.py`: Script de Fine-Tuning a 4-bits.
  * `resultados.py`: Inferencia y generación de predicciones.
  * `evaluar_metricas.py`: Cálculo de Accuracy y F1-Score.

## Hardware Utilizado
* Modelos Base: Google Colab
* Llama-3: Local (NVIDIA RTX 4070 8GB)
