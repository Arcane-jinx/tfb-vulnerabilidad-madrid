# tfb-vulnerabilidad-madrid

**Trabajo Final de Bàtxelor**  
*Análisis de la Vulnerabilidad Territorial y Desigualdad Sociodemográfica en los Distritos de la Ciudad de Madrid mediante Datos Abiertos y Aprendizaje Automático.*

## Contenido
- `interfaz_fase2_tfb.py`: interfaz / script principal.
- Datos: `informe_bi.xlsx`, `informe_ee.xlsx`, `panel_indicadores_distritos_barrios.csv`, `resultados_mejorados.xlsx`.
- Modelo: `modelo_vulnerabilidad_mejorado.pkl`.

---

## 📌 Descripción
Este repositorio contiene el desarrollo completo del Trabajo Final de Bàtxelor.  
El objetivo es analizar la **vulnerabilidad territorial** y la **desigualdad sociodemográfica** en los 21 distritos de Madrid utilizando **datos abiertos** y **técnicas de aprendizaje automático** (PCA, K-Means, Random Forest, SHAP).

El análisis se divide en dos fases:
- **Fase I** → Exploración inicial de datos abiertos y cálculo de un índice preliminar de vulnerabilidad.  
- **Fase II** → Modelado avanzado con técnicas de Machine Learning para identificar factores causales y realizar predicciones.  

---

## 📂 Estructura del repositorio

- `interfaz_fase2_tfb.py` → Script principal de la interfaz.  
- `notebooks/TFB_Vulnerabilidad_Madrid.ipynb` → Notebook con **todo el código utilizado en las fases I y II**.  
- `datos/` → Conjuntos de datos abiertos:  
  - `informe_bi.xlsx`  
  - `informe_ee.xlsx`  
  - `panel_indicadores_distritos_barrios.csv`  
  - `resultados_mejorados.xlsx`  
- `modelo_vulnerabilidad_mejorado.pkl` → Modelo entrenado final.  

---

## ▶️ Ejecución en Google Colab

Desde el notebook (`notebooks/TFB_Vulnerabilidad_Madrid.ipynb`) puedes abrir el proyecto directamente en **Google Colab** y replicar todo el análisis:

1. Haz clic en la pestaña **“Open in Colab”** en la parte superior del notebook.  
2. Sube los archivos de datos (`.csv` y `.xlsx`) cuando sea requerido.  
3. Ejecuta las celdas paso a paso para reproducir el flujo completo del TFB.  

---

## 📊 Resultados principales

- **Índice Compuesto de Vulnerabilidad (ICV)** para los 21 distritos.  
- Identificación de una **brecha norte-sur** en la ciudad de Madrid.  
- Factores clave: **educación** y **renta media** como determinantes principales de la vulnerabilidad.  
- Simulación de escenarios de intervención con impacto estimado sobre el ICV.  

---

✍️ *Autor*: **Anabel Ruiz Lorca**  
🎓 Universitat Carlemany – Bàtxelor en Ciencia de Datos
