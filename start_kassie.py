from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig  # Import PEFT classes
import torch
import gradio as gr

# Load the BASE model and tokenizer
base_model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16,  # Match the adapter's dtype
)

# Load the PEFT ADAPTER
peft_model_path = "./"
peft_model = PeftModel.from_pretrained(
    base_model,
    peft_model_path,
    adapter_name="kassie_adapter",
)

# Merge the adapter into the base model (optional but recommended for inference)
merged_model = peft_model.merge_and_unload()

# Create a text generation pipeline
qa_pipeline = pipeline(
    "text-generation",
    model=merged_model,  # Use the merged model
    tokenizer=tokenizer,
)

# Define a function to generate answers
def generate_answer(question):
    answer = qa_pipeline(
        question,
        max_length=512,
        temperature=0.7,  # Add sampling parameters
    )
    return answer[0]['generated_text']

# Create a Gradio interface
interface = gr.Interface(
    fn=generate_answer,
    inputs="text",
    outputs="text",
    title="Fine-Tuned Model Q&A",
)
interface.launch()
