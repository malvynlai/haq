import platform

arch = platform.machine()
sys = platform.system()
processor = platform.processor()
print(f"{arch}\n{sys}\n{processor}")

import onnxruntime as ort
import os
import numpy as np
import time
import gc

from pathlib import Path
from tokenizers import Tokenizer


def run_recipe_generation(ingredient_list: str) -> str:

    root_dir = Path.cwd()
    onnx_root = Path(ort.__file__).parent


    # Subdirectory where all .onnx dependencies are located
    model_subdirectory = "qnn-deepseek-r1-distill-qwen-1.5b"

    # The embeddings model is entry point, use netron to visualize
    model_name = "deepseek_r1_1_5_embeddings_quant_v2.0.onnx"

    # This graph is used to process initial prompt, we can pass up to 64 tokens
    context_model = "deepseek_r1_1_5_ctx_v2.1.onnx_ctx.onnx"

    # This graph is used to perform next word inference after the initial prompt
    context_model_iter = "deepseek_r1_1_5_iter_v2.1.onnx_ctx.onnx"

    # This graph allows us to take hidden states and return logits
    head_model = "deepseek_r1_1_5_head_quant_v2.0.onnx"

    # Tokenizer
    tokenizer_json = "tokenizer.json"

    model_path = root_dir/"models"/model_subdirectory/model_name
    ctx_path = root_dir/"models"/model_subdirectory/context_model
    ctx_path_itr = root_dir/"models"/model_subdirectory/context_model_iter
    head_path = root_dir/"models"/model_subdirectory/head_model
    tokenizer_path = root_dir/"models"/model_subdirectory/tokenizer_json
    hexagon_driver = onnx_root/"capi"/"QnnHtp.dll"

    session_options = ort.SessionOptions()

    qnn_provider_options = {
        # Path to the backend driver "Hexagon"
        "backend_path": hexagon_driver,
        # https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html#configuration-options
        "htp_performance_mode": "burst",
        "soc_model": "60",
        # "enable_htp_context_cache": "0",
        # "profiling_level": "detailed",
        # "profiling_file_path": root_dir/"models"/model_subdirectory/"profiling_deepseek_7b.csv",
        # Enabling graph optimization causes problems, need to look into this
        "htp_graph_finalization_optimization_mode": "3",
        "qnn_context_priority":"high",
    }


    embedding_session = ort.InferenceSession(model_path,
                                    providers= [("QNNExecutionProvider",qnn_provider_options)],
                                sess_options= session_options
                                )


    # Creating an inference session for the initial context graph
    ctx_session = ort.InferenceSession(ctx_path,
                                        providers=[("QNNExecutionProvider",qnn_provider_options)],
                                        sess_options= session_options
                                            )


    # Creating an inference session for the single prediction context graph (iter_ctx)
    ctx_itr_session = ort.InferenceSession(ctx_path_itr,
                                            providers=[("QNNExecutionProvider",qnn_provider_options)],
                                            sess_options= session_options
                                        )


    # Creating an inference session for the head session which will provide logits from hidden states
    head_session = ort.InferenceSession(head_path,
                                    providers= [("QNNExecutionProvider",qnn_provider_options)],
                                sess_options= session_options
                                )


    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # ingredients_prompt = "<|User|>\nI have the following items available: " + ", ".join(ingredient_list) + "."

    query = f"<|User|>\nPlease provide a simple recipe with these items: " + ", ".join(ingredient_list) + ". Pick only food related items.\n<|Assistant|><think>\n"

    # query = "<|User|>\nImagine you are a chef. Can you tell me a dish recipe with detailed instructions.\n<|Assistant|><think>\n"
    encoding = tokenizer.encode(query)
    input_ids = encoding.ids
    input_ids = np.array([input_ids], dtype=np.int64)
    embedding_output = embedding_session.run(None, {"input_ids": input_ids})[0]

    # Number of input sequences processed simultaneously
    batch_size = 1

    # Current sequence length for initial prompt (number of tokens in current sequence)
    seq_len = embedding_output.shape[1]

    # Dimensionality of each token embedding vector
    hidden_size = embedding_output.shape[2]

    # Number of attention heads in each transformer layer
    num_heads = 12

    # Size of each attention head (should be hidden_size // num_heads
    attn_head_size = 128 

    # Total number of transformer layers
    num_layers = 28

    # SWA
    max_seq_len = 64

    # Number of key/value heads (key/value heads are shared amongst attention heads)
    num_key_value_heads = 2


    # Let's initialize our KV cache for all transformer layers
    empty_kv = {}
    for i in range(num_layers):
        # Shape of key and value tensors for each transformer layer
        past_shape = (batch_size, num_key_value_heads, max_seq_len, attn_head_size)

        # Initialize past keys for layer i (used in attention mechanism to avoid recomputation
        empty_kv[f"past_keys_{i}"] = np.zeros(past_shape, dtype=np.float32)

        # Initialize past values for layer i
        empty_kv[f"past_values_{i}"] = np.zeros(past_shape, dtype=np.float32)


    # Subtract 1 to get the index of the last token in the sequence (since indexing is 0-based)
    init_sequence_length = np.array(embedding_output.shape[1]-1, dtype=np.int32).reshape(1,1)

    # Set the maximum sequence length for the model's current forward pass
    max_seq_length = np.array([max_seq_len], dtype=np.int32)

    seq_lens = {
        "past_seq_len": init_sequence_length,
        "total_seq_len": max_seq_length 
    }

    # pad the inputs to expected size of prefill graph
    batch_size, seq_len, embed_dim = embedding_output.shape
    padding_id = 151643
    padded_embedding = np.full((batch_size, max_seq_length[0], embed_dim), padding_id, dtype=embedding_output.dtype)
    padded_embedding[:, :seq_len, :] = embedding_output 

    prefill_inputs = {
        **empty_kv,
        **seq_lens,
        "input_hidden_states": padded_embedding,
    }

    prompt_outputs = ctx_session.run(None, prefill_inputs)
    output_hidden_states = prompt_outputs[0]

    present_kv = {f"past_keys_{i}": prompt_outputs[1 + i * 2] for i in range(num_layers)}
    present_kv.update({f"past_values_{i}": prompt_outputs[1 + i * 2 + 1] for i in range(num_layers)})

    logits = head_session.run(None, {"output_hidden_states": output_hidden_states})[0]

    def softmax_numpy(x: np.array, temperature: float=1) -> np.array:
        # stabilize x in case of large numbers 
        x = x - np.max(x, axis=-1, keepdims=True)

        # Apply temperature
        x = x/temperature

        # Apply Softmax
        return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)


    def top_k_probas(probas: np.array, k: int=5) -> np.array:
        # Copy probas so in-place operations don't work on original variable
        probas = probas.copy()
        # Normalize probabilities
        probas /= np.sum(probas)
        # Using -probas to get in descending order
        top_indices_sorted = np.argsort(-probas)[:k]
        top_k_probas = probas[top_indices_sorted]

        # Renormalize top-k probabilites to sum to 1 (probabilites must sum to 1 to use np.random.choice
        top_k_probas /= np.sum(top_k_probas)

        # Return top k probabilities
        return top_indices_sorted, top_k_probas


    def apply_repetition_penalty(logits, generated_ids, penalty=1.1):
        for token_id in set(generated_ids):
            logits[token_id] /= penalty
        return logits


    softmax = lambda x, temperature=1: np.exp((x-np.max(x, axis=-1, keepdims=True))/temperature)/np.sum(np.exp((x-np.max(x, axis=-1, keepdims=True))/temperature), axis=-1, keepdims=True)
    temp = 0.7
    probas = softmax(logits[0,-1], temperature=temp)

    next_token_id = int(np.random.choice(len(probas), p=probas))

    start = time.time()
    max_tokens = 2000
    top_k = 2
    generated_ids = [next_token_id]
    prev_seq_len = 64

    print("\nInitial Query:\n", query)
    print("Generated:")

    for _ in range(max_tokens):
        input_ids = np.array([[next_token_id]], dtype=np.int64)
        print(tokenizer.decode([next_token_id], skip_special_tokens=True),end="")
        embedding_output = embedding_session.run(None, {"input_ids": input_ids})[0]
        
        lengths = {
        "past_seq_len": np.array([[prev_seq_len]], dtype=np.int32),
        "total_seq_len": np.array([prev_seq_len + 1], dtype=np.int32)
        }

        iter_inputs = {
        "input_hidden_states": embedding_output,
        **present_kv,
        **lengths,
        }

        iter_outputs = ctx_itr_session.run(None, iter_inputs)

        output_hidden_states = iter_outputs[0]

        present_kv = {f"past_keys_{i}": iter_outputs[1 + i * 2] for i in range(num_layers)}
        present_kv.update({f"past_values_{i}":iter_outputs[1 + i * 2 + 1] for i in range(num_layers)})
        
        logits = head_session.run(None, {"output_hidden_states": output_hidden_states})[0]    

        token_logits = logits[0,-1]
        token_logits = apply_repetition_penalty(token_logits, generated_ids, penalty=1.1)
        # Get probabilities
        probas = softmax(token_logits, temperature=temp)
        top_indices, top_probas = top_k_probas(probas, k=top_k) 
        next_token_id = int(np.random.choice(top_indices, p=top_probas)) #int(np.argmax(probas))
        generated_ids.append(next_token_id)
        prev_seq_len += 1

        if next_token_id == tokenizer.token_to_id("<ï½œendâ–ofâ–sentenceï½œ>"):
            break
            
    end = time.time()
    elapsed = end - start
    tps = np.round((max_tokens / elapsed), 2)
    print(f"\nTokens Per Second: {tps}")
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output_text

if __name__=='__main__':
    run_recipe_generation(['chicken'])