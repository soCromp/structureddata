import os
import sys 
import argparse
import pandas as pd 
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import re 

# Add the parent directory to sys.path so we can import handler.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from handler import UnifiedDataLoader 

def serialize_row(row, columns):
    """Converts a pandas Series into a clean JSON string."""
    # convert all values to strings to prevent JSON serialization errors with dates/floats
    row_dict = {col: str(row[col]) for col in columns}
    return json.dumps(row_dict)

def parse_generated_text(text, columns):
    """Attempts to parse the LLM's JSON output."""
    parsed_row = {col: None for col in columns}
    try:
        # start = text.find('{')
        # end = text.rfind('}') + 1
        match = re.search(r'\{.*?\}', text, re.DOTALL) # non greedy
        if match is not None:
            json_str =match.group(0)
            data = json.loads(json_str)
            for k, v in data.items():
                if k in columns:
                    parsed_row[k] = v
        else: 
            print(f"\n[DEBUG] JSON Parse Failed!\nMatch was none\nRaw LLM Output:\n{text}\n{'-'*40}")
    except json.JSONDecodeError as e:
        print(f"\n[DEBUG] JSON Parse Failed!\nError: {e}\nRaw LLM Output:\n{text}\n{'-'*40}")
        pass 
        
    return parsed_row

def build_prompt(train_df, columns, k_shots=5):
    """Builds a few-shot prompt using JSON-formatted examples."""
    prompt = (
        "You are a tabular data synthesizer. Generate a single, realistic, and unique new data record "
        "that follows the exact JSON format and statistical distributions of the examples below.\n"
        "Output ONLY a single valid JSON object. Do not output any other text or formatting.\n\n"
    )
    
    samples = train_df.sample(k_shots)
    for _, row in samples.iterrows():
        prompt += serialize_row(row, columns) + "\n"
        
    print(prompt)
        
    prompt += "\nNow generate exactly one new record in the exact same JSON format:\n"
    # seed the generation with the opening brace to force JSON mode
    prompt += "{"
    return prompt

def main(args):
    print(f"Loading data for {args.dataset}...")
    loader = UnifiedDataLoader(dataset_name=args.dataset, target_model_type="llm")
    train_df = loader.get_train_data()
    meta = loader.get_metadata()
    columns = meta['columns']

    print(f"Initializing LLM: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # generation requires left-padding
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id
    )

    print(f"Pre-building {args.num_samples} prompts...")
    prompts = [build_prompt(train_df, columns, k_shots=args.k_shots) for _ in range(args.num_samples)]

    synthetic_rows = []
    print(f"Generating {args.num_samples} synthetic rows using batch size {args.batch_size}...")
    
    out_batches = generator(
        prompts, 
        max_new_tokens=1500,
        temperature=0.6, 
        do_sample=True,
        return_full_text=False,
        batch_size=args.batch_size
    )
    
    for output in tqdm(out_batches, total=args.num_samples):
        raw_text = output[0]['generated_text'].strip()
        
        # reattach the seeded opening brace
        full_generated_text = f"{{{raw_text}" 
        
        parsed_row = parse_generated_text(full_generated_text, columns)
        synthetic_rows.append(parsed_row)

    synth_df = pd.DataFrame(synthetic_rows)
    synth_df = synth_df[columns] 
    
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'synth', args.dataset))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "icl.csv")
    
    synth_df.to_csv(save_path, index=False)
    print(f"\nSuccessfully saved {len(synth_df)} synthetic records to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Few-Shot ICL Tabular Generation")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of rows to generate")
    parser.add_argument("--k_shots", type=int, default=5, help="Number of in-context examples")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B", help="HuggingFace Model ID")
    parser.add_argument("--max_tokens", type=int, default=250, help="Max tokens to generate per row")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of prompts to process simultaneously")
    
    args = parser.parse_args()
    main(args)
    