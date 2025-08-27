import os
import json
import yaml
from tqdm import tqdm
from config.config_wrapper import get_config_wrapper

def read_yaml(config='default'):
    if os.path.exists(f'config/prompt/{config}.yaml'):
        yaml_file = f'config/prompt/{config}.yaml'
    else:
        yaml_file = config
    with open(yaml_file, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

def read_yaml_mode(config='default', code_mode='noncode'):
    if code_mode == 'noncode' or code_mode == 'noncode_nothink':
        yaml_file = f'config/noncode_yaml/{config}.yaml'
    elif code_mode == 'pot' or code_mode == 'pot_nothink':
        yaml_file = f'config/python_yaml/{config}.yaml'
    elif code_mode == 'sandbox' or code_mode == 'sandbox_nothink': 
        yaml_file = f'config/sandbox_yaml/{config}.yaml'
    elif code_mode == 'agent' or code_mode == 'agent_nothink':
        yaml_file = f'config/agent_yaml/{config}.yaml'
    else:
        raise ValueError(f"Invalid code_mode: {code_mode}")
    with open(yaml_file, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)
    
def write_jsonl_lines(file, data):
    config_wrapper = get_config_wrapper()
    if config_wrapper.save_prompt:
        json.dump(data, file, ensure_ascii=False)
    else:
        data.pop(config_wrapper.prompt_key)
        json.dump(data, file, ensure_ascii=False)
    file.write('\n')
    file.flush()

def print_info(info):
    print('-'*100)
    print("[INFO] model_name:", info['model_name'])
    print("[INFO] splits:", info['splits'])
    print("[INFO] modes:", info['modes'])
    print("[INFO] output_dir:", info['output_dir'])
    print("[INFO] Infer Limit:", "No limit" if info['infer_limit'] is None else info['infer_limit'])
    print("[INFO] Number of Workers:", info['num_workers'])
    print("[INFO] Batch Size:", info['batch_size'])
    print("[INFO] Use Accel:", info['use_accel'])
    print('-'*100)

def read_json_or_jsonl(data_path, split='', mapping_key=None):
    base_path = os.path.join(data_path, split)
    if os.path.exists(f'{base_path}.json'):
        file_path = f'{base_path}.json'
    elif os.path.exists(f'{base_path}.jsonl'):
        file_path = f'{base_path}.jsonl'
    elif os.path.exists(f'{base_path}.jsonl.json'):
        file_path = f'{base_path}.jsonl.json'
    elif base_path.endswith('.json') or base_path.endswith('.jsonl') or base_path.endswith('.jsonl.json'):
        file_path = base_path
    else:
        print(f"base_path: {base_path}")
        raise FileNotFoundError("No JSON or JSONL file found.")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            if file_path.endswith('.json'):
                data = json.load(file)
            elif file_path.endswith('.jsonl'):
                data = []
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON on line {line_num} in {file_path}: {e}")
                        continue
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file {file_path}: {e}")
        raise
    except Exception as e:
        print(f"Error: Failed to read file {file_path}: {e}")
        raise
    
    if mapping_key:
        return {item[mapping_key]: item for item in data if mapping_key in item}
    else:
        return data

def read_json_or_jsonl_with_idx(data_path, split='', idx=None):
    base_path = os.path.join(data_path, split)
    if os.path.exists(f'{base_path}.json'):
        file_path = f'{base_path}.json'
    elif os.path.exists(f'{base_path}.jsonl'):
        file_path = f'{base_path}.jsonl'
    elif base_path.endswith('.json') or base_path.endswith('.jsonl'):
        file_path = base_path
    else:
        raise FileNotFoundError("No JSON or JSONL file found.")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            if file_path.endswith('.json'):
                data = json.load(file)
            elif file_path.endswith('.jsonl'):
                data = []
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON on line {line_num} in {file_path}: {e}")
                        continue
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file {file_path}: {e}")
        raise
    except Exception as e:
        print(f"Error: Failed to read file {file_path}: {e}")
        raise
    
    if idx is not None:
        try:
            return next(item for item in data if item.get('idx') == idx)
        except StopIteration:
            raise ValueError(f"No entry found for idx {idx}")
    else:
        return data