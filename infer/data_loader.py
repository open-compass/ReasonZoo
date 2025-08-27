import json
import os
from utils.common import read_yaml_mode, read_json_or_jsonl, read_json_or_jsonl_with_idx

# Load the data
def load_data(split='', mode='', code_mode='noncode'):
    if split in ['communication_code', 'number_calculation', 'gradeschoolmath', 'formal_language', 'operation_research', 'puzzle_and_code','physics','dailylogic','boolean_logic'] and mode in ['zero-shot']:
        rule = read_json_or_jsonl(f'data/{split}', 'rule', 'idx')
        print(f"doing {split} {mode} for rule")
        sample = read_json_or_jsonl(f'data/{split}', 'sample')
        print(f"doing {split} {mode} for sample")
        config = f'{mode}'
        if mode == 'think':
            config = 'zero-shot'
        template = read_yaml_mode(config, code_mode)
        for s in sample:
            rule_id = s['rule_id']
            rule_content = rule[rule_id]['rule_content']
            question = s['question']

            if config in ['zero-shot', 'zero-shot-cot']:
                prompt_format = [rule_content, question]
                prompt = template[f'{split}_prompt_format'][0].format(*prompt_format)

            s['title'] = rule[rule_id].get('title', '')
            s['tag'] = rule[rule_id].get('tag', '')
            s['rule_content'] = rule_content
            yield prompt, s
    
if __name__ == '__main__':
    last_prompt = None

    for prompt, sample in load_data('cipher', 'subquestions'):
        last_prompt = prompt
        # print(sample)

    if last_prompt is not None:
        print(last_prompt)
        

