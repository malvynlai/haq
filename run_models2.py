import onnxruntime as ort
from pathlib import Path
from pipeline import main


LLMPath = ''
tokenizer = ''

def format_prompt(labels):
    label_str = ', '.join(labels)
    return f"Given the following ingredients: {label_str}, write a simple recipe using them. Ignore non-food items. Assume we have basic cooking ingredients like salt, pepper, and oil. Return your response in markdown format."

def tokenize(prompt, tokenizer):
    inputs = tokenizer(prompt, return_tensors="np", padding=True)
    return inputs["input_ids"]

def decode_output(output_ids, tokenizer):
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def run_mistral_onnx(prompt, model_path, tokenizer):
    qnn_provider_options = {
        'backend_path': str(Path(ort.__file__).parent / 'capi' / 'QnnHtp.dll'),
    }

    session = ort.InferenceSession(str(model_path),
                                   providers=[("QNNExecutionProvider", qnn_provider_options), "CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    input_ids = tokenize(prompt, tokenizer)

    outputs = session.run([output_name], input_ids)
    return decode_output(outputs, tokenizer)

def main(filepath):
    mistral_onnx_path = Path(LLMPath) 
    labels = pipeline.main(filepath)
    # labels = ['apple', 'banana', 'carrot']
    prompt = format_prompt(labels)
    recipe = run_mistral_onnx(prompt, LLMPath, tokenizer)

    print("Generated Recipe:\n", recipe)

if __name__ == "__main__":
    main('fridge.jpg')
