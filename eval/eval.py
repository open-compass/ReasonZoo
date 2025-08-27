from eval_utils import evaluate_responses, extract_decision_from_judge_response
from utils.common import read_json_or_jsonl
import os
import json
import csv
import sys
import argparse
from prettytable import PrettyTable

SPLITS = ["graph", "communication_code", "puzzle_and_code", "number_calculation", "gradeschoolmath", "operation_research", "physics", "dailylogic", "boolean_logic", "formal_language", "phybench", "math500", "aime24", "aime25","livemathbench", "gpqa"]

def get_question_type_and_mode(filename):
    """
    Determines the question type and mode from a filename by checking for substrings.

    Args:
        filename (str): The filename to parse.

    Returns:
        tuple: A tuple containing the question type (str) and mode (str).
    """
    question_type = None
    for split in SPLITS:
        if split in filename:
            question_type = split
            break
            
    parts = os.path.basename(filename).split('_')
    mode = parts[-1].replace('.jsonl', '')
        
    return question_type, mode

def evaluate_all_files_in_folder(folder_path, output_folder, csv_file, use_llm_judge=False, api_key=None, base_url=None, max_workers=8, tasks_to_judge=None, model_path='Qwen/Qwen2.5-72B-Instruct'):
    """
    Evaluate all files in a folder and generate a summary CSV file.
    
    Args:
        folder_path: Path to folder containing JSONL files to evaluate
        output_folder: Path to save evaluation results
        csv_file: Path to save CSV summary
        use_llm_judge: Whether to use LLM-based judge for evaluation
        api_key: API key for LLM service
        base_url: Base URL for LLM service
        max_workers: Maximum number of parallel workers for LLM evaluation
        tasks_to_judge: List of tasks to use LLM judge for (defaults to ['logic'])
    """
    if tasks_to_judge is None:
        tasks_to_judge = ['boolean_logic', 'physics']
    if not os.path.exists(output_folder) and output_folder != "":
        os.makedirs(output_folder, exist_ok=True)
    model_scores = {}
    question_type = None
    mode = None
    failed_files = []  # Track failed files for summary
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            print(f"Processing {filename}...")
            try:
                parts = os.path.basename(filename).split('_')
                model_name = parts[0]
                question_type, mode = get_question_type_and_mode(filename)

                print(question_type, mode)
                
                # Try to read the input data file
                try:
                    data = read_json_or_jsonl(folder_path, filename)
                    if not data:
                        print(f"Warning: {filename} contains no data, skipping...")
                        continue
                except Exception as e:
                    print(f"Error: Failed to read {filename}: {e}")
                    failed_files.append(f"{filename}: Failed to read file - {str(e)}")
                    continue
                
                # Determine if we should use LLM-based evaluation for this file
                should_use_llm_judge = use_llm_judge
                if should_use_llm_judge:
                    print(f"Using LLM-based judge for {question_type} task evaluation")
                    
                output_file = os.path.join(output_folder, f"evaluation_{filename}.json")
                
                # Check if output file already exists and merge existing results
                existing_data = []
                if os.path.exists(output_file):
                    try:
                        # Fix: Split the path and filename for proper read_json_or_jsonl call
                        output_folder_path = os.path.dirname(output_file)
                        # keep the 2.5 format
                        output_filename = os.path.basename(output_file)
                        existing_data = read_json_or_jsonl(output_folder_path, output_filename)
                        print(f"Found existing output file {output_file} with {len(existing_data)} entries")
                        
                        # Check if we should skip because file is complete and recent
                        if (len(existing_data) == len(data) and 
                            not ("deepseek" in filename.lower() or "qwen3" in filename.lower())):
                            # Check if most entries have valid judge responses
                            judge_response_count = 0
                            for entry in existing_data:
                                can_reuse, _ = extract_decision_from_judge_response(
                                    entry.get('judge_response', '') or 
                                    entry.get('LLM_response', '') or 
                                    entry.get('llm_response', '')
                                )
                                if can_reuse or entry.get('is_correct') is not None:
                                    judge_response_count += 1
                            
                            reuse_ratio = judge_response_count / len(existing_data) if existing_data else 0
                            if reuse_ratio > 0.8:  # If >80% have judge responses, skip
                                print(f"Skipping evaluation for {filename} because {judge_response_count}/{len(existing_data)} entries have judge responses")
                                continue
                            else:
                                print(f"Will merge with existing data: {judge_response_count}/{len(existing_data)} entries have judge responses")
                        
                    except Exception as e:
                        print(f"Error reading output file {output_file}: {e}")
                        print(f"Will re-evaluate {filename}")
                        existing_data = []

                # Merge existing data with input data by idx
                merged_data = []
                existing_by_idx = {str(item.get('idx', '')): item for item in existing_data}
                
                for input_item in data:
                    input_idx = str(input_item.get('idx', ''))
                    if input_idx in existing_by_idx:
                        # Merge: use existing data but update with any new fields from input
                        merged_item = existing_by_idx[input_idx].copy()
                        # Update with any new fields from input data, but preserve existing judge responses
                        for key, value in input_item.items():
                            if key not in ['judge_response', 'LLM_response', 'llm_response', 'is_correct']:
                                merged_item[key] = value
                        merged_data.append(merged_item)
                    else:
                        # New item, add as-is
                        merged_data.append(input_item)
                
                # Use merged data for evaluation
                data_to_evaluate = merged_data
                
                # Try to evaluate the responses
                try:
                    evaluation_results = evaluate_responses(
                        data_to_evaluate, 
                        question_type, 
                        mode, 
                        use_llm_judge=should_use_llm_judge,
                        api_key=api_key,
                        base_url=base_url,
                        max_workers=max_workers,
                        model_path=model_path
                    )
                except Exception as e:
                    print(f"Error: Failed to evaluate {filename}: {e}")
                    failed_files.append(f"{filename}: Failed during evaluation - {str(e)}")
                    continue

                # Try to write the output file
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    print(f"Error: Failed to write output file {output_file}: {e}")
                    failed_files.append(f"{filename}: Failed to write output - {str(e)}")
                    continue

                # Handle different formats of is_correct (can be boolean/int or list)
                correct_count = 0
                for result in evaluation_results:
                    if isinstance(result['is_correct'], list):
                        # If is_correct is a list, count it as correct if all elements are truthy
                        # or if the list has any truthy elements (depending on your requirements)
                        correct_count += 1 if any(result['is_correct']) else 0
                    else:
                        # If is_correct is a boolean or int
                        correct_count += result['is_correct']
                        
                count = len(evaluation_results)
                accuracy = (correct_count / count) * 100 if count > 0 else 0

                
                # Store results in a nested dictionary for each model and mode
                key = (model_name, mode)
                if key not in model_scores:
                    model_scores[key] = {}
                model_scores[key][question_type] = {
                    'correct': correct_count,
                    'total': count,
                    'accuracy': accuracy,
                }

                # Print individual file results
                print(f"Processed {filename}: Total Correct - {correct_count} out of {count}, Accuracy - {accuracy:.2f}%")

            except Exception as e:
                print(f"Error: Unexpected error processing {filename}: {e}")
                failed_files.append(f"{filename}: Unexpected error - {str(e)}")
                continue

    # Print summary of failed files
    if failed_files:
        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Successfully processed: {len([f for f in os.listdir(folder_path) if f.endswith('.jsonl')]) - len(failed_files)} files")
        print(f"Failed to process: {len(failed_files)} files")
        print("Failed files:")
        for failed_file in failed_files:
            print(f"  - {failed_file}")
        print("=" * 50)
    else:
        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Successfully processed all {len([f for f in os.listdir(folder_path) if f.endswith('.jsonl')])} files")
        print("=" * 50)

    # Aggregate results and write to CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['model_name', 'mode', 'total_correct', 'total_count', 'overall_accuracy']
        question_types = set(qt for scores in model_scores.values() for qt in scores)
        for qt in sorted(question_types):
            fieldnames.extend([f'{qt}_correct', f'{qt}_total', f'{qt}_accuracy'])
        print(fieldnames)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        table = PrettyTable()
        table.field_names = fieldnames

        for (model_name, mode), scores in model_scores.items():
            total_correct = sum(details['correct'] for details in scores.values())
            total_count = sum(details['total'] for details in scores.values())
            overall_accuracy = (total_correct / total_count) * 100 if total_count > 0 else 0
            if mode == 'mixed':
                total_pass_rate = sum(details['pass_rate'] for details in scores.values()) / len(scores)
                overall_pass_rate = sum(details['pass_rate'] for details in scores.values()) / len(scores) if len(scores) > 0 else 0
            row = {
                'model_name': model_name,
                'mode': mode,
                'total_correct': total_correct,
                'total_count': total_count,
                'overall_accuracy': f"{overall_accuracy:.2f}%"
            }
            if mode == 'mixed':
                row['overall_pass_rate'] = f"{overall_pass_rate:.2f}%"

            for question_type, details in scores.items():
                row[f'{question_type}_correct'] = details['correct']
                row[f'{question_type}_total'] = details['total']
                row[f'{question_type}_accuracy'] = f"{details['accuracy']:.2f}%"
            print(row)
            writer.writerow(row)
            try:
                table.add_row([row[field] for field in fieldnames])
            except Exception as e:
                print(f"Error adding row to table: {e}")
            # Print summarized results
            print(f"Model: {model_name}, Mode: {mode}, Total Correct: {total_correct}, Total: {total_count}, Overall Accuracy: {overall_accuracy:.2f}%" )
    print(table)

if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description='Evaluate JSONL files and generate a summary CSV file.')

    # Basic arguments
    parser.add_argument('source_folder', type=str, help='Path to the folder containing JSONL files for evaluation.')
    parser.add_argument('target_root_folder', type=str, help='Path to the folder where output JSON files and the CSV will be stored.')
    parser.add_argument('csv_file', type=str, help='Path to the output CSV file that will store the aggregated results.')
    
    # LLM evaluation arguments
    parser.add_argument('--use_llm_judge', action='store_true', help='Use LLM-based judge for evaluation')
    parser.add_argument('--api_key', type=str, default=os.getenv("OPENAI_API_KEY"), help='API key for the LLM service')
    parser.add_argument('--base_url', type=str, default=os.getenv("OPENAI_API_BASE_URL"), help='Base URL for the LLM service')
    parser.add_argument('--max_workers', type=int, default=8, help='Maximum number of parallel workers for LLM evaluation')
    parser.add_argument('--tasks_to_judge', nargs='+', default=['physics', 'boolean_logic'], help='Tasks to use LLM judge for')
    parser.add_argument('--model_path', type=str, default='gpt-4.1', help='Model path for the LLM service')
    # Parse arguments
    args = parser.parse_args()

    # Call the function with these parameters
    evaluate_all_files_in_folder(
        args.source_folder, 
        args.target_root_folder, 
        args.csv_file,
        use_llm_judge=args.use_llm_judge,
        api_key=args.api_key,
        base_url=args.base_url,
        max_workers=args.max_workers,
        tasks_to_judge=args.tasks_to_judge,
        model_path=args.model_path
    )


