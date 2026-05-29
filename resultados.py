import os
os.environ["PYTHONUTF8"] = "1"

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- MISMAS CONFIGURACIONES DE ANTES ---
model_id = "NousResearch/Meta-Llama-3-8B"
adapter_path = "./mi_llama_womenhelp"
test_file = "test.csv" 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 2. CARGAR DATOS Y DETECTAR COLUMNA
df_test = pd.read_csv(test_file)

# Buscamos cuál es la columna que contiene el texto (usualmente 'TEXT', 'text' o la primera columna)
posibles_nombres = ['TEXT', 'text', 'Text', 'reporte', 'REPORT']
columna_texto = None

for nombre in posibles_nombres:
    if nombre in df_test.columns:
        columna_texto = nombre
        break

if columna_texto is None:
    # Si no encontramos ninguno de esos nombres, tomamos la primera columna que sea texto
    columna_texto = df_test.columns[0]

print(f"✅ Usando la columna '{columna_texto}' para el análisis.")

class_map = {0: "leve", 1: "medio", 2: "alto", 3: "grave"}
inv_class_map = {v: k for k, v in class_map.items()}

results = []

# 3. PROCESO DE INFERENCIA
print(f"🚀 Clasificando {len(df_test)} reportes...")

with torch.no_grad():
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        # Accedemos dinámicamente a la columna detectada
        contenido = str(row[columna_texto])
        
        prompt = f"Analiza la gravedad: {contenido} \n\nGravedad:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=2,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        respuesta = full_text.split("Gravedad:")[-1].strip().lower()
        
        pred_id = 1 # Default
        for label_str, label_id in inv_class_map.items():
            if label_str in respuesta:
                pred_id = label_id
                break
                
        results.append(pred_id)

# 4. GUARDAR
df_test['CLASS'] = results
df_test.to_csv("predicciones_competicion.csv", index=False)

print("\n✅ ¡Archivo 'predicciones_competicion.csv' generado con éxito!")