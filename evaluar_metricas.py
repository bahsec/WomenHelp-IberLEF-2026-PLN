import os
os.environ["PYTHONUTF8"] = "1"

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("🚀 Iniciando evaluación de métricas para el paper...")

# 1. Configuración de modelo en 4-bits para la RTX 4070
model_id = "NousResearch/Meta-Llama-3-8B"
adapter_path = "./mi_llama_womenhelp"
eval_file = "devel.csv" # Usamos el set de validación que SÍ tiene etiquetas

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

print("Cargando el modelo...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 2. Cargar datos de validación
df_eval = pd.read_csv(eval_file)
class_map = {0: "leve", 1: "medio", 2: "alto", 3: "grave"}
inv_class_map = {v: k for k, v in class_map.items()}

y_true = df_eval['CLASS'].tolist() # Estas son las respuestas correctas
y_pred = [] # Aquí guardaremos las predicciones del modelo

print(f"✅ Evaluando {len(df_eval)} reportes. Esto tomará unos minutos...")

# 3. Proceso de Inferencia
with torch.no_grad():
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval)):
        texto = str(row['TEXT'])
        prompt = f"Analiza la gravedad: {texto} \n\nGravedad:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=2,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        respuesta = full_text.split("Gravedad:")[-1].strip().lower()
        
        pred_id = 1 # Por defecto
        for label_str, label_id in inv_class_map.items():
            if label_str in respuesta:
                pred_id = label_id
                break
                
        y_pred.append(pred_id)

# 4. GENERAR MÉTRICAS PARA EL PAPER
print("\n" + "="*50)
print("📊 RESULTADOS OFICIALES PARA EL PAPER 📊")
print("="*50)

# Accuracy General
acc = accuracy_score(y_true, y_pred)
print(f"\nExactitud (Accuracy) General: {acc * 100:.2f}%\n")

# Reporte detallado (Precision, Recall, F1-Score)
print("--- REPORTE DE CLASIFICACIÓN (F1-SCORE) ---")
nombres_clases = ["0 (Leve)", "1 (Medio)", "2 (Alto)", "3 (Grave)"]

# try-except por si falta la clase 3 en las predicciones
try:
    reporte = classification_report(y_true, y_pred, target_names=nombres_clases, zero_division=0)
    print(reporte)
except Exception as e:
    print(classification_report(y_true, y_pred, zero_division=0))

print("--- MATRIZ DE CONFUSIÓN ---")
print(confusion_matrix(y_true, y_pred))
print("="*50)
print("✅ Copia estos resultados y envíaselos a tu equipo.")