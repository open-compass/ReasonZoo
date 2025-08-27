from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from utils.build_conversation import build_conversation
from config.config_wrapper import config_wrapper
import re
import os
import tempfile
import subprocess
import requests
import json
import ast
import textwrap
from black import format_file_contents, FileMode

def load_model(model_name, model_args, use_accel=False, code_mode='noncode'):
    model_path = model_args.get('model_name')
    tp = model_args.get('tp', 8)
    model_components = {}
    model_components['code_mode'] = code_mode
    if use_accel:
        model_components['use_accel'] = True
        model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if 'DeepSeek-V2' in model_name:
            model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, max_model_len=8192, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True)
        else:
            model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True)
        model_components['model_name'] = model_name
    else:
        model_components['use_accel'] = False
        model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_components['model'] = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
        model_components['model_name'] = model_name
    return model_components

def extract_python_scripts(text):
    """
    Extracts all Python code snippets from the text.

    Args:
        text (str): The text containing Python code.

    Returns:
        List[str]: A list of extracted Python code snippets.
    """
    # Define both types of markers
    start_markers = ["'''python", "```python"]
    end_markers = ["'''", "```"]

    snippets = []

    # Iterate over both types of markers
    for start_marker, end_marker in zip(start_markers, end_markers):
        start_indices = [i for i in range(len(text)) if text.startswith(start_marker, i)]
        for start in start_indices:
            # Find the corresponding end marker after this start marker
            end = text.find(end_marker, start + len(start_marker))
            if end != -1:
                snippets.append(text[start + len(start_marker):end].strip())

    return snippets

def is_safe_code(code):
    """
    Checks if the provided Python code is safe to execute.
    
    Args:
        code (str): The Python code to check.
        
    Returns:
        bool: True if the code is considered safe, False otherwise.
    """
    # Define a list of potentially dangerous imports and functions
    dangerous_imports = [
        'os.system', 'subprocess', 'shutil.rmtree', 'sys.exit', 
        'eval(', 'exec(', '__import__', 'importlib', 
        'open(', 'file(', 'Shell', 'pty', 'socket', 'requests'
    ]
    
    # Check for dangerous imports or functions
    for dangerous_import in dangerous_imports:
        if dangerous_import in code:
            return False
    
    # Block any attempts to write to files
    if 'open(' in code and 'w' in code:
        return False
    
    # Additional safety checks can be added here
    
    return True

def execute_python_code(code, time_limit=10):
    """
    Executes the provided Python code and extracts the output (stdout).

    Args:
        code (str): The Python code to execute.
        time_limit (int): Maximum time allowed for code execution in seconds.

    Returns:
        tuple: A tuple containing the printed output (str) and the return code (int).
    """
    # First check if the code is safe to execute
    if not is_safe_code(code):
        return "⚠️ Code execution blocked for security reasons. The code contains potentially unsafe operations.", 1
    
    # Check if code contains main() function but doesn't have if __name__ == '__main__':
    has_main = 'def main' in code
    has_main_guard = 'if __name__ == ' in code
    
    # Only modify the code if there's a main function without the main guard
    if has_main and not has_main_guard:
        modified_code = code + """

if __name__ == '__main__':
    result = main()
    if result is not None:
        print(f"Return value: {result}")
"""
    else:
        modified_code = code

    # Create a temporary Python script file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(modified_code)
        temp_script_path = temp_file.name

    # Check if the code was written successfully
    if not os.path.exists(temp_script_path):
        return "Failed to create the temporary script file.", 1

    try:
        # Run the script with a timeout
        result = subprocess.run(
            ["python", temp_script_path],
            capture_output=True,
            text=True,
            timeout=time_limit
        )
        # Return the output and the exit code
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return f"Execution exceeded the time limit of {time_limit} seconds.", 1
    except Exception as e:
        return str(e), 1
    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)

def extract_python_blocks(message: str) -> list[str]:
    """Return *only* well-formed Python code blocks."""
    CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    blocks = CODE_BLOCK_RE.findall(message)
    cleaned: list[str] = []
    for raw in blocks:
        code = textwrap.dedent(raw).strip()
        # quick sanity check: can the code be parsed?
        try:
            ast.parse(code, mode="exec")
        except SyntaxError:
            continue          # skip this block – it's not valid Python
        cleaned.append(code)
    return cleaned

def prettify(code: str) -> str:
    """Format with Black so indentation & spacing are always valid."""
    try:
        return format_file_contents(code, fast=True, mode=FileMode())
    except Exception:
        return code          # fall back to original if Black blows up

def infer(prompts, historys, **kwargs):
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer', None)
    model_name = kwargs.get('model_name', None)
    use_accel = kwargs.get('use_accel', False)
    use_budget_forcing = kwargs.get('use_budget_forcing', False)
    max_tokens_thinking = kwargs.get('max_tokens_thinking', 32000)
    code_mode = kwargs.get('code_mode', 'noncode')
    print(f"DEBUG: code_mode: {code_mode}")
    if code_mode == 'sandbox':
        use_sandbox = True
    else:
        use_sandbox = False

    if isinstance(prompts[0], str):
        messages = [build_conversation(history, prompt) for history, prompt in zip(historys, prompts)]
    else:
        raise ValueError("Invalid prompts format")
    
    if use_accel:
        if use_budget_forcing and not use_sandbox:
            responses = []
            for message in messages:
                try:
                    # First apply chat template to get the prompt text (not token ids)
                    prompt_text = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors=None)
                    if not isinstance(prompt_text, str):
                        # Some tokenizers return tensors, convert to string if needed
                        prompt_text = tokenizer.decode(prompt_text)
                    
                    print(f"DEBUG: Chat template applied, prompt length: {len(prompt_text)}")
                    print(f"DEBUG: Prompt start: {prompt_text[:100]}...")
                    
                    # Add thinking marker directly to the text prompt
                    thinking_prompt = prompt_text + "<|im_start|>think"
                    print(f"DEBUG: Added thinking marker: {thinking_prompt[-20:]}")
                    
                    # Get stop tokens for thinking phase
                    thinking_stop_tokens = ["<|im_start|>", "<|im_end|>"]
                    stop_token_ids_thinking = []
                    for token in thinking_stop_tokens:
                        ids = tokenizer.encode(token, add_special_tokens=False)
                        if isinstance(ids, list):
                            stop_token_ids_thinking.extend(ids)
                        else:
                            stop_token_ids_thinking.append(ids)
                    
                    # Try to also detect model-specific stop tokens
                    model_type = model_name.lower()
                    if 'llama' in model_type:
                        # Add Llama-specific stop tokens
                        additional_stops = ["<s>", "</s>"]
                        for token in additional_stops:
                            try:
                                ids = tokenizer.encode(token, add_special_tokens=False)
                                if isinstance(ids, list):
                                    stop_token_ids_thinking.extend(ids)
                                else:
                                    stop_token_ids_thinking.append(ids)
                            except:
                                pass
                    elif 'qwen' in model_type:
                        # Add Qwen-specific stop tokens
                        additional_stops = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
                        for token in additional_stops:
                            try:
                                ids = tokenizer.encode(token, add_special_tokens=False)
                                if isinstance(ids, list):
                                    stop_token_ids_thinking.extend(ids)
                                else:
                                    stop_token_ids_thinking.append(ids)
                            except:
                                pass
                    
                    print(f"DEBUG: Stop token IDs for thinking: {stop_token_ids_thinking}")
                    
                    # Initial thinking phase
                    sampling_params_thinking = SamplingParams(
                        max_tokens=max_tokens_thinking,
                        min_tokens=0,
                        stop_token_ids=stop_token_ids_thinking,
                        skip_special_tokens=False,
                        temperature=0.0,
                    )
                    
                    print(f"DEBUG: Starting thinking phase with max tokens: {max_tokens_thinking}")
                    thinking_output = model.generate(
                        prompts=[thinking_prompt],
                        sampling_params=sampling_params_thinking
                    )
                    
                    print(f"DEBUG: Thinking output length: {len(thinking_output[0].outputs[0].text)}")
                    print(f"DEBUG: Thinking output start: {thinking_output[0].outputs[0].text[:100]}...")
                    
                    # Store initial thinking text
                    initial_thinking_text = thinking_output[0].outputs[0].text
                    
                    # Extract and execute Python code from initial thinking
                    python_snippets = extract_python_scripts(initial_thinking_text)
                    code_execution_results = []
                    
                    for i, snippet in enumerate(python_snippets):
                        print(f"DEBUG: Executing Python snippet {i+1} of {len(python_snippets)}")
                        output, return_code = execute_python_code(snippet)
                        execution_status = "SUCCESS" if return_code == 0 else "ERROR"
                        code_execution_results.append({
                            "snippet": snippet,
                            "output": output,
                            "status": execution_status
                        })
                    
                    # Full prompt with initial thinking
                    full_prompt = thinking_prompt + thinking_output[0].outputs[0].text
                    max_tokens_thinking_tmp = max_tokens_thinking
                    
                    # Store additional thinking text
                    additional_thinking_text = ""
                    
                    # Handle ignore phases if needed
                    if max_tokens_thinking_tmp > 0:
                        ignore_str = "Wait"
                        
                        # If we executed code, add the results before the "Wait" marker
                        if code_execution_results:
                            code_results_text = "\n\nCODE EXECUTION RESULTS:\n"
                            for i, result in enumerate(code_execution_results):
                                code_results_text += f"\n--- Snippet {i+1} ({result['status']}) ---\n"
                                code_results_text += f"{result['output']}\n"
                            
                            # Add code execution results to the prompt
                            full_prompt += code_results_text
                            ignore_str = "\n" + ignore_str
                        
                        for i in range(100):
                            # Reduce remaining thinking budget
                            tokens_used = len(thinking_output[0].outputs[0].token_ids)
                            max_tokens_thinking_tmp -= tokens_used
                            print(f"DEBUG: Ignore phase {i+1}, tokens used: {tokens_used}, remaining budget: {max_tokens_thinking_tmp}")
                            
                            full_prompt += ignore_str
                            print(f"DEBUG: Added ignore string: {full_prompt[-10:]}")
                            
                            # Continue thinking with reduced budget
                            sampling_params_thinking = SamplingParams(
                                max_tokens=max_tokens_thinking_tmp,
                                min_tokens=1,
                                stop_token_ids=stop_token_ids_thinking,
                                skip_special_tokens=False,
                                temperature=0.0,
                            )
                            
                            thinking_output = model.generate(
                                prompts=[full_prompt],
                                sampling_params=sampling_params_thinking
                            )
                            
                            print(f"DEBUG: Additional thinking output length: {len(thinking_output[0].outputs[0].text)}")
                            print(f"DEBUG: Additional thinking output start: {thinking_output[0].outputs[0].text[:100]}...")
                            
                            # Store additional thinking text
                            additional_thinking_text += thinking_output[0].outputs[0].text
                            
                            # Extract and execute Python code from additional thinking
                            additional_snippets = extract_python_scripts(thinking_output[0].outputs[0].text)
                            additional_code_execution_results = []
                            
                            if additional_snippets:
                                print(f"DEBUG: Found {len(additional_snippets)} Python snippets in additional thinking")
                                for j, snippet in enumerate(additional_snippets):
                                    print(f"DEBUG: Executing additional Python snippet {j+1} of {len(additional_snippets)}")
                                    output, return_code = execute_python_code(snippet)
                                    execution_status = "SUCCESS" if return_code == 0 else "ERROR"
                                    result = {
                                        "snippet": snippet,
                                        "output": output,
                                        "status": execution_status
                                    }
                                    additional_code_execution_results.append(result)
                                    code_execution_results.append(result)
                                
                                # Add code execution results to the prompt
                                if additional_code_execution_results:
                                    code_results_text = "\n\nADDITIONAL CODE EXECUTION RESULTS:\n"
                                    for j, result in enumerate(additional_code_execution_results):
                                        code_results_text += f"\n--- Additional Snippet {j+1} ({result['status']}) ---\n"
                                        code_results_text += f"{result['output']}\n"
                                    
                                    full_prompt += code_results_text
                            
                            full_prompt += thinking_output[0].outputs[0].text
                    
                    # Final answer phase
                    stop_token_ids = [tokenizer.eos_token_id]
                    if 'Meta-Llama-3' in model_name:
                        stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
                    
                    # Add more model-specific stop tokens
                    if 'llama' in model_type:
                        try:
                            stop_token_ids.append(tokenizer.convert_tokens_to_ids("</s>"))
                        except:
                            pass
                    
                    print(f"DEBUG: Final answer phase, stop tokens: {stop_token_ids}")
                    
                    # Add final answer marker if not already present
                    if "Final Answer:" not in full_prompt and "final answer:" not in full_prompt.lower():
                        # Before adding final answer marker, add a summary of all code execution
                        if code_execution_results:
                            full_prompt += "\n\nSUMMARY OF ALL CODE EXECUTION RESULTS:\n"
                            for i, result in enumerate(code_execution_results):
                                is_additional = i >= len(code_execution_results) - len(additional_code_execution_results) if 'additional_code_execution_results' in locals() else False
                                snippet_type = "Additional" if is_additional else "Initial"
                                full_prompt += f"\n--- {snippet_type} Snippet {i+1} ({result['status']}) ---\n"
                                full_prompt += f"{result['output']}\n"
                        
                        full_prompt += "\nFinal Answer: "
                    
                    # Create sampling params without stop tokens to prevent early cutoff
                    sampling_params_final = SamplingParams(
                        max_tokens=config_wrapper.max_tokens,
                        # No stop tokens to allow complete generation
                    )
                    
                    final_output = model.generate(
                        prompts=[full_prompt],
                        sampling_params=sampling_params_final
                    )
                    
                    final_text = final_output[0].outputs[0].text
                    print(f"DEBUG: Final output length: {len(final_text)}")
                    print(f"DEBUG: Final output: {final_text[:100]}...")
                    
                    # If the response is empty or very short, try once more with a more explicit prompt
                    if len(final_text.strip()) < 5:
                        print(f"DEBUG: Response was too short, trying again with explicit prompt")
                        explicit_prompt = full_prompt + "\nPlease provide the final answer in the required format: "
                        
                        final_output = model.generate(
                            prompts=[explicit_prompt],
                            sampling_params=sampling_params_final
                        )
                        
                        final_text = final_output[0].outputs[0].text
                        print(f"DEBUG: New final output length: {len(final_text)}")
                        print(f"DEBUG: New final output: {final_text[:100]}...")
                    
                    # Include thinking parts in the response
                    # Full response includes thinking and final answer
                    full_response = f"INITIAL THINKING:\n{initial_thinking_text}"
                    
                    # Include initial code execution results if any
                    initial_results = code_execution_results
                    additional_results = []
                    
                    if 'additional_code_execution_results' in locals() and additional_code_execution_results:
                        additional_results = additional_code_execution_results
                        initial_results = code_execution_results[:len(code_execution_results)-len(additional_code_execution_results)]
                    
                    if initial_results:
                        code_results_text = "\n\nINITIAL CODE EXECUTION RESULTS:\n"
                        for i, result in enumerate(initial_results):
                            code_results_text += f"\n--- Initial Snippet {i+1} ({result['status']}) ---\n"
                            code_results_text += f"{result['output']}\n"
                        full_response += code_results_text
                    
                    if additional_thinking_text:
                        full_response += f"\n\nADDITIONAL THINKING AFTER WAIT:\n{additional_thinking_text}" 
                        
                        # Include additional code execution results if any
                        if additional_results:
                            code_results_text = "\n\nADDITIONAL CODE EXECUTION RESULTS:\n"
                            for i, result in enumerate(additional_results):
                                code_results_text += f"\n--- Additional Snippet {i+1} ({result['status']}) ---\n"
                                code_results_text += f"{result['output']}\n"
                            full_response += code_results_text
                    
                    full_response += f"\n\nFINAL ANSWER:\n{final_text}"
                    responses.append(full_response)
                except Exception as e:
                    print(f"DEBUG ERROR in budget forcing: {str(e)}")
                    # Fallback to standard generation
                    prompt_text = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors=None)
                    if not isinstance(prompt_text, str):
                        prompt_text = tokenizer.decode(prompt_text)
                    
                    # Add explicit prompt for the required format
                    if "Determine whether the following formula is" in prompt_text:
                        prompt_text += "\nPlease provide your answer in the required format."
                    
                    stop_token_ids = [tokenizer.eos_token_id]
                    if 'Meta-Llama-3' in model_name:
                        stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
                    
                    model_type = model_name.lower()
                    if 'llama' in model_type:
                        try:
                            stop_token_ids.append(tokenizer.convert_tokens_to_ids("</s>"))
                        except:
                            pass
                    
                    print(f"DEBUG: Using fallback with stop tokens: {stop_token_ids}")
                    
                    sampling_params = SamplingParams(
                        max_tokens=config_wrapper.max_tokens,
                        # Remove stop tokens from fallback as well
                        temperature=0.2  # Slight temperature may help prevent empty responses
                    )
                    
                    output = model.generate(
                        prompts=[prompt_text],
                        sampling_params=sampling_params
                    )
                    
                    final_text = output[0].outputs[0].text
                    print(f"DEBUG: Fallback output length: {len(final_text)}")
                    print(f"DEBUG: Fallback output: {final_text[:100]}...")
                    
                    # If still empty, try with a more explicit system message
                    if len(final_text.strip()) < 5:
                        print(f"DEBUG: Fallback response too short, trying with explicit system message")
                        
                        # Try to extract user message and modify system message
                        if "<|im_start|>user" in prompt_text:
                            user_part = prompt_text.split("<|im_start|>user")[1]
                            if "<|im_end|>" in user_part:
                                user_message = user_part.split("<|im_end|>")[0]
                                
                                # Create new message with stronger system instruction
                                new_message = [
                                    {"role": "system", "content": "You must always provide a direct, concise answer. For logic problems, analyze step by step and then provide the final result in the exact format requested."},
                                    {"role": "user", "content": user_message.strip()}
                                ]
                                
                                modified_prompt = tokenizer.apply_chat_template(new_message, add_generation_prompt=True, return_tensors=None)
                                if not isinstance(modified_prompt, str):
                                    modified_prompt = tokenizer.decode(modified_prompt)
                                
                                # For the modified system prompt fallback, also remove stop tokens
                                modified_sampling_params = SamplingParams(
                                    max_tokens=config_wrapper.max_tokens,
                                    temperature=0.2  # Slight temperature may help prevent empty responses
                                )
                                
                                output = model.generate(
                                    prompts=[modified_prompt],
                                    sampling_params=modified_sampling_params
                                )
                                
                                final_text = output[0].outputs[0].text
                                print(f"DEBUG: Modified fallback output length: {len(final_text)}")
                                print(f"DEBUG: Modified fallback output: {final_text[:100]}...")
                    
                    # Include thinking parts in the response
                    # Full response includes thinking and final answer
                    full_response = f"INITIAL THINKING:\n{initial_thinking_text}"
                    if additional_thinking_text:
                        full_response += f"\n\nADDITIONAL THINKING AFTER WAIT:\n{additional_thinking_text}" 
                    full_response += f"\n\nFINAL ANSWER:\n{final_text}"
                    responses.append(full_response)
        
        elif use_sandbox:
            """
            Single-stream loop with safeguards:
            1. Build one prompt string from the current message list.
            2. Generate until the closing code marker ("\n```\n") or .
            3. Append the assistant's response to the message list.
            4. Extract the latest code block from the response.
            5. Run only NEW python blocks in SandboxFusion, avoiding re-execution.
            6. Append execution results + cue as new messages to the list.
            7. Repeat, tracking errors and breaking on repetition or limits.
            """
            print(f"DEBUG: Using sandbox with message list management")
            # Maximum number of rounds to iterate
            max_rounds = 8  # Adjust as needed
            MAX_SAME_ERROR = 2  # Max times to retry same failing code
            MAX_PROMPT_TOKENS = 30000 # Safety break based on token count estimate

            # Track executed code to avoid redundant runs
            import hashlib
            import textwrap

            def digest(code):
                """Create stable identifier for code snippets (ignoring whitespace)"""
                code = textwrap.dedent(code).strip()
                return hashlib.sha1(code.encode()).hexdigest()

            responses = []
            print(f"DEBUG: messages: {messages}")
            for prompt_idx, initial_msg_list in enumerate(messages):
                # Work on a copy to avoid modifying the original input
                current_msg_list = [msg.copy() for msg in initial_msg_list]
                print(f"DEBUG: Processing message list {prompt_idx}, initial length: {len(current_msg_list)}")

                # Setup tracking variables for de-duplication and loop control
                executed_snippets = {}  # {digest: (stdout, stderr, success)}
                already_seen_blocks = set()  # set of digests
                error_counter = {}  # {digest: count_of_consecutive_failures}
                prev_code_digest = None # Track digest of the previously executed block
                try:
                    for round_num in range(max_rounds):
                        print(f"DEBUG: Round {round_num} of {max_rounds}")

                        # --- Prepare Prompt for this Round ---
                        # Apply chat template to the *current* conversation history
                        # Add generation prompt to cue the model for a response
                        prompt_str_for_round = tokenizer.apply_chat_template(
                            current_msg_list, add_generation_prompt=True, return_tensors=None
                        )
                        if not isinstance(prompt_str_for_round, str):
                             # Decode if the template returns token IDs
                             prompt_str_for_round = tokenizer.decode(prompt_str_for_round)

                        # Estimate token count (approximation) and check limit
                        # A more accurate method would involve tokenizing prompt_str_for_round
                        estimated_tokens = len(prompt_str_for_round) // 4 # Rough estimate
                        print(f"DEBUG: Estimated tokens for round {round_num}: {estimated_tokens}")
                        if estimated_tokens > MAX_PROMPT_TOKENS:
                             print(f"DEBUG: Estimated tokens ({estimated_tokens}) exceeded limit ({MAX_PROMPT_TOKENS}), breaking loop.")
                             break

                        # --- Generate Next Segment ---
                        sampling_params = SamplingParams(
                            max_tokens=4096, # Tokens for *this* generation step
                            temperature=0.8,
                            stop=["\n```\n", "</s>", "<|im_end|>"] # Stop after code or at EOS
                        )

                        new_text = "" # Initialize new_text for the round
                        stop_reason = None
                        try:
                            print(f"DEBUG: Calling model.generate with prompt (estimated tokens: {estimated_tokens})...")
                            raw_outputs = model.generate(prompts=[prompt_str_for_round],
                                                       sampling_params=sampling_params)

                            if raw_outputs and isinstance(raw_outputs, list) and len(raw_outputs) > 0:
                                if hasattr(raw_outputs[0], 'outputs') and len(raw_outputs[0].outputs) > 0:
                                    output_data = raw_outputs[0].outputs[0]
                                    new_text = output_data.text
                                    stop_reason = output_data.finish_reason
                                    print(f"DEBUG: Model generated {len(new_text)} chars, stop_reason: {stop_reason}")
                                else:
                                    print(f"DEBUG: Unexpected output structure in raw_outputs[0]: {raw_outputs[0]}")
                            else:
                                print(f"DEBUG: Unexpected output format or empty output: {raw_outputs}")

                        except Exception as e:
                            print(f"DEBUG: Error during model generation: {str(e)}")
                            # Add error as a message and break
                            current_msg_list.append({"role": "user", "content": f"Error generating response: {str(e)}"})
                            break

                        # Check if we got an empty response
                        if not new_text or new_text.strip() == "":
                            print("DEBUG: Empty response, breaking loop")
                            break

                        # --- Append Assistant Response to History ---
                        # Add the raw model output as an assistant message
                        current_msg_list.append({"role": "assistant", "content": new_text})
                        print(f"DEBUG: Appended assistant message. current_msg_list length: {len(current_msg_list)}")

                                                # Check if we hit max length limit
                        if stop_reason == 'length':
                            print(f"DEBUG: Model stopped due to max length. Requesting final answer.")
                            # Add a message requesting a final, concise answer
                            current_msg_list.append({
                                "role": "user", 
                                "content": "Your response was cut off due to length limits. Now directly give your answer in FINAL ANSWER format:"
                            })
                            # Continue to next round to get the final answer
                            continue

                        # Check if a non-code stop sequence was hit
                        hit_eos_stop = stop_reason == 'stop' and any(
                            new_text.endswith(s) for s in sampling_params.stop if s != "\n```\n"
                        )
                        if hit_eos_stop:
                             print(f"DEBUG: Model stopped due to EOS token: {stop_reason}. Ending sandbox loop.")
                             # The final assistant message is already added.
                             break # Exit the loop, no more code expected

                        # --- Code Extraction and Execution ---
                        code_to_execute = None
                        current_code_digest = None
                        # Find the start of the last python code block in the *newly generated text*
                        code_start_marker = "```python"
                        code_start_index = new_text.rfind(code_start_marker)

                        if code_start_index != -1:
                            # Extract code from the start marker to the end of new_text
                            # (The model stopped at "\n```\n", so new_text ends just before the closing marker)
                            code = new_text[code_start_index + len(code_start_marker):].strip()

                            # The stop sequence "\n```\n" was consumed by the generator,
                            # but we need it for proper markdown structure in the history.
                            # Add it back to the assistant's message content.
                            current_msg_list[-1]["content"] += "\n```\n"
                            print(f"DEBUG: Appended closing code marker to assistant message.")


                            if code: # Ensure extracted code is not empty
                                code_to_execute = code
                                current_code_digest = digest(code_to_execute)

                                # Check for repeated code block
                                if current_code_digest == prev_code_digest and round_num > 0:
                                     print(f"DEBUG: Model repeated the same code block (digest: {current_code_digest}). Breaking loop.")
                                     # Add a note to the history? Maybe just break.
                                     current_msg_list.append({"role": "user", "content": "The model repeated the previous code block. Stopping interaction."})
                                     break
                                prev_code_digest = current_code_digest # Update tracker

                                # Check for previously seen block
                                if current_code_digest in already_seen_blocks:
                                     print(f"DEBUG: Skipping already seen code block (digest: {current_code_digest}).")
                                     # Add a message indicating skip and cue for new code
                                     cue_msg = {
                                         "role": "user",
                                         "content": "This code block was already attempted. Let's try a different approach."
                                     }
                                     current_msg_list.append(cue_msg)
                                     continue # Skip execution, go to next generation round

                                already_seen_blocks.add(current_code_digest)

                                # --- Execute the new code block ---
                                execution_result_msg = None
                                try:
                                    print(f"DEBUG: Executing new snippet (digest: {current_code_digest}):\n{code_to_execute}")
                                    formatted_snippet = prettify(code_to_execute) # Assuming prettify exists

                                    res = requests.post('http://localhost:8080/run_code', json={
                                        'code': formatted_snippet,
                                        'language': 'python',
                                    })
                                    res.raise_for_status()
                                    res_json = res.json()

                                    run_result = res_json.get('run_result', {})
                                    stdout = run_result.get('stdout', '')
                                    stderr = run_result.get('stderr', '')
                                    success = res_json.get('status') == 'Success'

                                    executed_snippets[current_code_digest] = (stdout, stderr, success)

                                    # Format result block content
                                    result_content = "```output\n" # Start output block
                                    if success:
                                        error_counter[current_code_digest] = 0
                                        result_content += f"{stdout}" if stdout else "Execution successful (no stdout)."
                                    else:
                                        if len(stderr) > 1000:
                                            stderr = stderr[-1000:]
                                        error_counter[current_code_digest] = error_counter.get(current_code_digest, 0) + 1
                                        result_content += f"--- Sandbox ERROR ---\n{stderr}"
                                        if error_counter[current_code_digest] >= MAX_SAME_ERROR:
                                             result_content += (
                                                 f"\n\nThe sandbox has seen this exact error {error_counter[current_code_digest]} times. "
                                                 f"Let's try a different approach."
                                             )
                                    result_content += "\n```\nLet's continue based on this output." # End output block (no final newline needed inside content)

                                    # Create the message for the execution result
                                    # Using 'user' role to represent the sandbox output back to the assistant
                                    execution_result_msg = {"role": "user", "content": result_content}


                                except Exception as e:
                                    print(f"DEBUG: Error during sandbox execution or result processing: {str(e)}")
                                    if current_code_digest:
                                        executed_snippets[current_code_digest] = ('', str(e), False)
                                        error_counter[current_code_digest] = error_counter.get(current_code_digest, 0) + 1
                                    # Create an error message
                                    error_content = f"```output\n--- Sandbox Execution ERROR ---\n{str(e)}\n```"
                                    execution_result_msg = {"role": "user", "content": error_content}

                                # --- Append Execution Result and Cue to History ---
                                if execution_result_msg:
                                    current_msg_list.append(execution_result_msg)
                                    print(f"DEBUG: Appended execution result message. current_msg_list length: {len(current_msg_list)}")

                            else:
                                # Code block marker found, but code was empty after stripping
                                print("DEBUG: Extracted code block was empty. Breaking loop.")
                                current_msg_list.append({"role": "user", "content": "The model provided an empty code block. Stopping interaction."})
                                break # Stop if model emits empty code block


                        else:
                            # No ```python marker found in the new_text.
                            # Model finished its response without generating code in this turn.
                            print("DEBUG: No python code block found in the latest generation. Ending sandbox loop.")
                            # The final assistant message is already in current_msg_list
                            break # Exit the loop

                    # --- End of Round Loop ---

                except Exception as e:
                    print(f"DEBUG: Error in sandbox processing loop for message list {prompt_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Add error message to the history for this specific input
                    current_msg_list.append({"role": "user", "content": f"Error during sandbox processing: {str(e)}"})

                # --- Finalize Response for this Input ---
                # Convert the final message list back into a single string using the template
                # Do not add generation prompt here, we want the final state.
                final_prompt_str = tokenizer.apply_chat_template(
                    current_msg_list, add_generation_prompt=False, return_tensors=None
                )
                if not isinstance(final_prompt_str, str):
                    final_prompt_str = tokenizer.decode(final_prompt_str)

                responses.append(final_prompt_str)
                print(f"DEBUG: Finished processing message list {prompt_idx}. Final string length: {len(final_prompt_str)}")
        
        else:
            # Original implementation without budget forcing
            prompt_texts = []
            for message in messages:
                prompt_text = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors=None)
                if not isinstance(prompt_text, str):
                    # Some tokenizers return tensors, convert to string if needed
                    prompt_text = tokenizer.decode(prompt_text)
                prompt_texts.append(prompt_text)
            
            stop_token_ids = [tokenizer.eos_token_id]
            if 'Meta-Llama-3' in model_name:
                stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
            
            # For the standard implementation (non-budget forcing), also remove stop tokens
            sampling_params = SamplingParams(
                max_tokens=config_wrapper.max_tokens,
                # No stop tokens to allow complete generation
            )
            
            outputs = model.generate(
                prompts=prompt_texts,
                sampling_params=sampling_params
            )
            
            responses = []
            for output in outputs:
                response = output.outputs[0].text
                responses.append(response)
    else:
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, padding=True, truncation=True, return_dict=True, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=config_wrapper.max_tokens, do_sample=False)
        responses = []
        for i, prompt in enumerate(prompts):
            response = tokenizer.decode(outputs[i, len(inputs['input_ids'][i]):], skip_special_tokens=True)
            responses.append(response)

    return responses

if __name__ == '__main__':

    prompts = [
        '''Who are you?''',
        '''only answer with "I am a chatbot"''',
    ]
    model_args = {
        'model_name': '01-ai/Yi-1.5-6B-Chat',
        'model_type': 'local',
        'tp': 8
    }
    model_components = load_model("Yi-1.5-6B-Chat", model_args, use_accel=True)
    # Example with budget forcing
    responses = infer(prompts, None, **model_components)
    for response in responses:
        print(response)
