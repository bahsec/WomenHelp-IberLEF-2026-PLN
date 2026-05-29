import os
os.environ["PYTHONUTF8"] = "1" # Solución para Unicode en Windows

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

print("🚀 Iniciando bypass ultra-estable para la RTX 4070...")

# 1. Cargar Datos
df_train = pd.read_csv('train.csv')
df_devel = pd.read_csv('devel.csv')
class_map = {0: "leve", 1: "medio", 2: "alto", 3: "grave"}

# 2. Tokenizer y Preparación
model_id = "NousResearch/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    prompts = []
    # Generamos los textos manualmente
    for text, label_idx in zip(examples['TEXT'], examples['CLASS']):
        p = f"Analiza la gravedad: {text} \n\nGravedad: {class_map[label_idx]} {tokenizer.eos_token}"
        prompts.append(p)
    
    return tokenizer(prompts, truncation=True, max_length=512, padding="max_length")

# FIX CRUCIAL: Convertimos los nombres de las columnas a una lista de Python real
cols_list = [str(c) for c in df_train.columns]

train_ds = Dataset.from_pandas(df_train).map(
    tokenize_function, 
    batched=True, 
    remove_columns=cols_list # <-- Aquí estaba el error, ahora es una lista pura
)
devel_ds = Dataset.from_pandas(df_devel).map(
    tokenize_function, 
    batched=True, 
    remove_columns=cols_list
)

# 3. Cargar Modelo con Cuantización (4-bits para tus 8GB VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 4. Configurar LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# 5. Configurar Entrenamiento
training_args = TrainingArguments(
    output_dir="./resultados_womenhelp",
    per_device_train_batch_size=2, # Mantener en 2 para no saturar memoria
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_steps=100,
    logging_steps=10,
    bf16=True,
    optim="paged_adamw_8bit",
    save_total_limit=1,
    report_to="none"
)

# 6. El Entrenador Estándar (El más fiable)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=devel_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("🔥 ¡Todo listo! Lanzando entrenamiento en la GPU...")
trainer.train()

# 7. Guardar el resultado final
model.save_pretrained("mi_llama_womenhelp")
print("✅ ¡ÉXITO! Entrenamiento terminado y modelo guardado.")