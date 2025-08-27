from openai import OpenAI
from models import model_configs
from utils.build_conversation import build_conversation
from config.config_wrapper import config_wrapper
from black import format_file_contents, FileMode, format_str
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAIError
import hashlib, textwrap, json, requests, re, sys
import uuid
import time

def load_model(model_name="", base_url="", api_key="", model="", call_type='api_chat', code_mode='noncode'):
    model_components = {}
    model_components['model_name'] = model_name
    model_components['model'] = model
    model_components['base_url'] = base_url
    model_components['api_key'] = api_key
    model_components['call_type'] = call_type
    model_components['code_mode'] = code_mode
    return model_components

@retry(
    stop=stop_after_attempt(50),
    wait=wait_exponential(multiplier=1, min=1, max=1000),
    retry=retry_if_exception_type(OpenAIError),
)
def request(messages, timeout=2000, max_tokens=8192, base_url="", api_key="", model="", model_name=None):
    client = OpenAI(base_url=base_url, api_key=api_key)
    if not model_name:
        try:
            model_name = client.models.list().data[0].id
        except Exception as e:
            print(f"Warning: Could not retrieve model name from API: {e}")
            model_name = model if model else "DeepSeek-V3-0324"
            print(f"Using fallback model name: {model_name}")
    else:
        model_name = model_name
    print(f"DEBUG: model_name: {model_name}")
    print(f"DEBUG: messages: {messages}")
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.8,
        top_p=0.8,
        stop=["</s>", "<|im_end|>"],
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return response

@retry(
    stop=stop_after_attempt(50),
    wait=wait_exponential(multiplier=1, min=1, max=1000),
    retry=retry_if_exception_type(OpenAIError),
)
def request_to_base_model(prompt, timeout=2000, max_tokens=2000, base_url="", api_key="", model="", model_name=None):
    
    client = OpenAI(base_url=base_url, api_key=api_key)
    if not model_name:
        try:
            model_name = client.models.list().data[0].id
        except Exception as e:
            print(f"Warning: Could not retrieve model name from API: {e}")
            model_name = model if model else "DeepSeek-V3-0324"
            print(f"Using fallback model name: {model_name}")
    print(f"DEBUG: model_name: {model_name}")
    response = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        timeout=timeout
    )
    print(response)
    
    return response

@retry(
    stop=stop_after_attempt(50),
    wait=wait_exponential(multiplier=1, min=1, max=1000),
    retry=retry_if_exception_type(OpenAIError),
)

def request_with_sandbox(
        messages,                          # full chat history – list[dict]
        timeout=120,
        max_tokens=8192,
        base_url="",
        api_key="",
        model="",
        model_name=None,
        sandbox_url="http://localhost:8080/run_code",
        max_rounds=8,
        enable_thinking=True,
    ):
    """
    Run an *interactive* loop in which the LLM can emit python code blocks.
    New blocks are executed in a remote SandboxFusion instance and the
    stdout/stderr are fed straight back into the conversation.

    Parameters
    ----------
    messages : list[dict]
        The running conversation in OpenAI chat-format.
    timeout : int
        Seconds allowed for the OpenAI request.
    max_tokens : int
        Tokens allotted per **generation** step (not for the full convo).
    sandbox_url : str
        POST endpoint that receives {'code': str, 'language': 'python'} and
        returns {'status': 'Success'|'Error', 'run_result': {'stdout': str,
        'stderr': str}}.

    Returns
    -------
    list[dict]
        The *augmented* conversation list containing the assistant's final
        answer plus any execution-result messages the loop generated.
    """
    print(f"DEBUG: base_url{base_url},api_key{api_key}")
    client = OpenAI(base_url=base_url, api_key=api_key)
    MAX_SAME_ERROR = 2  # Max times to retry same failing code
    MAX_PROMPT_TOKENS = 30000  # Safety limit (rough estimate)
    
    import hashlib, textwrap, json, requests, re, sys

    # Keep track of executed code and results
    executed_snippets = {}  # {digest: (stdout, stderr, success)}
    already_seen_blocks = set()  # set of digests
    error_counter = {}  # {digest: count of consecutive failures}
    prev_code_digest = None  # Track previously executed block
    
    print(f"Starting sandbox execution with {len(messages)} messages")

    def _digest(code: str) -> str:
        """Create stable identifier for code snippets (ignoring whitespace)"""
        return hashlib.sha1(textwrap.dedent(code).strip().encode()).hexdigest()
    
    def _estimate_tokens(messages):
        """Rough token count estimate based on characters"""
        total_chars = sum(len(m["content"]) if m.get("content") else 0 for m in messages)
        return total_chars // 4  # Rough approximation

    def prettify(code: str) -> str:
        """Format python code using black."""
        try:
            return format_str(code, mode=FileMode()).strip()
        except Exception as e:
            print(f"Warning: Black formatting failed: {e}. Using original code.", file=sys.stderr)
            return code

    for round_idx in range(max_rounds):
        print(f"Sandbox round {round_idx+1} of {max_rounds}")
        
        # Check token count to avoid excessive usage
        estimated_tokens = _estimate_tokens(messages)
        if estimated_tokens > MAX_PROMPT_TOKENS:
            print(f"Estimated tokens ({estimated_tokens}) exceeded limit ({MAX_PROMPT_TOKENS})")
            messages.append({
                "role": "user",
                "content": "The conversation has exceeded token limits. Please provide your final answer."
            })
            break
            
        # --- 1️⃣  Ask the model for the next step ---------------------------
        try:
            if not model_name:
                try:
                    current_model_name = client.models.list().data[0].id
                except Exception as e:
                    print(f"Warning: Could not retrieve model name from API: {e}")
                    current_model_name = model if model else "DeepSeek-V3-0324"
                    print(f"Using fallback model name: {current_model_name}")
            else:
                current_model_name = model_name
            print(f"DEBUG: model_name: {current_model_name}")
            print(f"DEBUG: messages: {messages}")
            resp = None
            if enable_thinking == False:
                resp = client.chat.completions.create(
                    model=current_model_name,
                    messages=messages,
                    temperature=0.8,
                    top_p=0.8,
                    stop=["</s>", "<|im_end|>"],
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                )
            else:
                resp = client.chat.completions.create(
                    model=current_model_name,
                    messages=messages,
                    temperature=0.8,
                    top_p=0.8,
                    stop=["</s>", "<|im_end|>"],
                    max_tokens=max_tokens,
                )
        except Exception as e:
            print(f"Error during OpenAI call: {e}", file=sys.stderr)
            messages.append({
                "role": "user",
                "content": f"Error during OpenAI call: {e}. Please provide your final answer based on what we have so far."
            })
            break

        assistant_reply = resp.choices[0].message.content or ""
        stop_reason = resp.choices[0].finish_reason
        messages.append({"role": "assistant", "content": assistant_reply})
        

        # --- 2️⃣  Extract python code block if present -----------------
        code_block = None
        current_digest = None

        # Look for python code blocks with improved extraction
        code_start_marker = "```python"
        code_start_index = assistant_reply.rfind(code_start_marker)

        if code_start_index != -1:
            # Get content after the start marker
            code_content = assistant_reply[code_start_index + len(code_start_marker):]
            
            # Check if there's an end marker in the extracted content
            code_end_index = code_content.find("```")
            
            if code_end_index != -1:
                # If end marker exists, extract text before it
                code_block = code_content[:code_end_index].strip()
            else:
                # No end marker found, take all content after start marker
                # (likely because generation stopped at end marker)
                code_block = code_content.strip()
                stop_reason = "stop"
                # Add closing marker back to message history if generation stopped there
                if stop_reason == "stop":
                    messages[-1]["content"] += "<im_end>"
                    print("Added closing code marker to assistant message")
            
            # Only proceed if we found valid code
            if code_block:
                current_digest = _digest(code_block)
                
                # Check for repeated or previously seen code blocks
                if current_digest == prev_code_digest and round_idx > 0:
                    print(f"Model repeated the same code block (digest: {current_digest})")
                    messages.append({
                        "role": "user",
                        "content": "You've repeated the previous code block. Please try a different approach or provide your final answer."
                    })
                    continue
                    
                # Check for previously seen block
                if current_digest in already_seen_blocks:
                    print(f"Skipping already seen code block (digest: {current_digest})")
                    messages.append({
                        "role": "user",
                        "content": "This code block was already attempted. Let's try a different approach."
                    })
                    continue
                
                # Mark this block as seen and update tracking
                already_seen_blocks.add(current_digest)
                prev_code_digest = current_digest
            else:
                print("Extracted empty code block, skipping execution")
                continue
        else:
            # No code block found
            print("No python code block found in this response")
            # Check if we're done (e.g., final answer provided)
            if "final answer" in assistant_reply.lower() or round_idx >= max_rounds - 1:
                break
            else:
                messages.append({
                    "role": "user", 
                    "content": "I was expecting a Python code block. Please provide your solution as a ```python code block."
                })
                continue

        # --- 3️⃣  Run formatted code in sandbox -----------------
        try:
            print(f"Executing code snippet (digest: {current_digest})")
            formatted_code = prettify(code_block) if code_block else ""
            res = requests.post(
                sandbox_url,
                json={"code": formatted_code, "language": "python"},
                timeout=timeout,
            )
            res.raise_for_status()
            res_json = res.json()

            status_ok = res_json.get("status") == "Success"
            run_res = res_json.get("run_result", {})
            stdout = run_res.get("stdout", "")
            stderr = run_res.get("stderr", "")
            
            executed_snippets[current_digest] = (stdout, stderr, status_ok)

            if status_ok:
                error_counter[current_digest] = 0
                result_text = stdout if stdout else "Execution finished with no stdout."
            else:
                # Truncate extremely long error messages
                if len(stderr) > 1000:
                    stderr = stderr[-1000:]
                    
                error_counter[current_digest] = error_counter.get(current_digest, 0) + 1
                result_text = f"--- Sandbox ERROR ---\n{stderr}"
                
                if error_counter[current_digest] >= MAX_SAME_ERROR:
                    result_text += (
                        f"\n\nThis exact error has occurred {error_counter[current_digest]} times. "
                        f"Let's try a completely different approach."
                    )

        except Exception as e:
            status_ok = False
            error_counter[current_digest] = error_counter.get(current_digest, 0) + 1
            executed_snippets[current_digest] = ("", str(e), False)
            result_text = f"--- Sandbox Execution ERROR ---\n{e}"

        # --- 4️⃣  Feed result back to the model -----------------------------
        messages.append({
            "role": "user",
            "content": f"```output\n{result_text}\n```\nLet's continue based on this output."
        })

    # 🔚 out-of-rounds or finished
    print(f"Sandbox execution completed after {round_idx+1} rounds")
    return messages


@retry(
    stop=stop_after_attempt(50),
    wait=wait_exponential(multiplier=1, min=1, max=1000),
    retry=retry_if_exception_type(OpenAIError),
)
def request_with_budget_forcing(
        messages,                          # full chat history – list[dict]
        timeout=2000,
        max_tokens_thinking=32000,
        max_output_tokens=8192,
        base_url="",
        api_key="",
        model="",
        model_name=None,
    ):
    """
    Run an API-based version of budget forcing that allows a model to "think" within a token limit,
    then produce a final answer. If the model doesn't finish its thinking in one go, it's prompted
    to continue with "Wait". If it exceeds the token limit, it's truncated and prompted for a final answer.

    Parameters
    ----------
    messages : list[dict]
        The running conversation in OpenAI chat-format.
    timeout : int
        Seconds allowed for the OpenAI request.
    max_tokens_thinking : int
        Maximum tokens to allocate for the thinking phase.
    max_output_tokens : int
        Maximum tokens for the final answer.

    """
    print(f"Starting budget forcing with max_tokens_thinking={max_tokens_thinking}, max_output_tokens={max_output_tokens}")
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    # Get actual model name if not specified
    if not model_name:
        try:
            model_name = client.models.list().data[0].id
        except Exception as e:
            print(f"Warning: Could not retrieve model name from API: {e}")
            model_name = model if model else "DeepSeek-V3-0324"
            print(f"Using fallback model name: {model_name}")
    print(f"DEBUG: model_name: {model_name}")
    
    # Make a copy of the original messages to avoid modifying the input
    messages_copy = messages.copy()
    
    # Add a system message at the start of the conversation to set expectations for thinking
    found_system = False
    for msg in messages_copy:
        if msg["role"] == "user" or msg["role"] == "system":
            found_system = True
            msg["content"] += "\nPlease think step by step. Your thinking will be interrupted if it gets too long, but you can continue from where you left off when prompted with 'Wait'."
            break
    
    if not found_system:
        # Add a new system message if none exists
        if messages_copy[0]["role"] == "user":
            messages_copy.insert(0, {
                "role": "user",
                "content": "Please think step by step. Your thinking will be interrupted if it gets too long, but you can continue from where you left off when prompted with 'Wait'."
            })
        else:
            messages_copy.insert(0, {
                "role": "system",
                "content": "Please think step by step. Your thinking will be interrupted if it gets too long, but you can continue from where you left off when prompted with 'Wait'."
            })
    
    # PHASE 1: THINKING PHASE
    thinking_responses = []
    remaining_tokens = max_tokens_thinking
    
    # Start the thinking process
    thinking_messages = messages_copy.copy()
    
    # Add an initial assistant message to start the thinking process
    thinking_messages.append({
        "role": "assistant",
        "content": "Let me think about this step by step:"
    })
    
    for thinking_round in range(100):  # Original + num_ignore continuation rounds
        if remaining_tokens <= 0:
            break
            
        try:
            print(f"Thinking round {thinking_round+1}, remaining tokens: {remaining_tokens}")
            response = client.chat.completions.create(
                model=model_name,
                messages=thinking_messages,
                max_tokens=min(remaining_tokens, 8192),  # API limit per request
                temperature=0.8,
                top_p=0.8,
                stop=["</s>", "<|im_end|>"],
                timeout=timeout
            )
            
            thinking_content = response.choices[0].message.content
            tokens_used = response.usage.completion_tokens
            remaining_tokens -= tokens_used
            
            thinking_responses.append(thinking_content)
            
            # If model finished before using all tokens or we're at the last round, force the model to continue
            
            # First, save what the assistant said
            thinking_messages.append({
                "role": "assistant",
                "content": thinking_content
            })
            # Replace the last stop token with 'wait'
            # 1. find the last stop token
            for stop_token in ["</s>", "<|im_end|>"]:
                if stop_token in thinking_content:
                    thinking_content = thinking_content.replace(stop_token, "Wait,")
                    break
                # if no stop token found, add 'Wait' to the end of the content
                if stop_token not in thinking_content:
                    thinking_content += "\nWait,"
            # remove the last line break
            thinking_content = thinking_content.rstrip("\n")
            
            # Then add the "Wait" message from user to force continuation
            thinking_messages.append({
                "role": "user",
                "content": "Please continue"
            })
            
            
        except Exception as e:
            print(f"Error during thinking phase: {e}")
            thinking_responses.append(f"Error occurred during thinking: {str(e)}")
            break
    
    # Combine all thinking responses with appropriate spacing
    combined_thinking = " ".join(thinking_responses)
    
    # PHASE 2: FINAL ANSWER PHASE
    # Create final answer prompt 
    final_messages = messages_copy.copy()
    
    # Now add all the thinking as assistant message, and append FINAL ANSWER:
    final_messages.append({
        "role": "assistant",
        "content": combined_thinking + "\n\nFINAL ANSWER:"
    })
    
    try:
        final_response = client.chat.completions.create(
            model=model_name,
            messages=final_messages,
            max_tokens=max_output_tokens,
            timeout=timeout
        )
        final_answer = final_response.choices[0].message.content
    except Exception as e:
        print(f"Error during final answer phase: {e}")
        final_answer = f"Error occurred during final answer: {str(e)}"
    
    # Construct complete response for the user
    full_response = combined_thinking + "\n\nFINAL ANSWER: " + final_answer
    
    # Return a special format containing both the full thinking + answer and just the answer
    return {"content": final_answer, "full_response": full_response}



# --- Start: New function combining budget forcing and sandbox ---
@retry(
    stop=stop_after_attempt(50),
    wait=wait_exponential(multiplier=1, min=1, max=1000),
    retry=retry_if_exception_type(OpenAIError),
)
def request_with_budget_forcing_and_sandbox(
        messages,                          # full chat history – list[dict]
        timeout=2000,
        max_tokens_thinking=32000,
        max_output_tokens=8192,
        base_url="",
        api_key="",
        model="",
        model_name=None,
        sandbox_url="http://localhost:8080/run_code",
        max_sandbox_rounds=100, # Limit sandbox executions within budget forcing
        enable_thinking=True,
    ):
    """
    Combines budget forcing with sandbox execution. The model thinks step-by-step,
    and any python code blocks generated during thinking are executed in a sandbox.
    The execution results are fed back for the next thinking step. Finally, a
    final answer is generated based on the entire thinking process including code execution.
    """
    print(f"Starting budget forcing with sandbox: max_tokens_thinking={max_tokens_thinking}, max_output_tokens={max_output_tokens}, sandbox_url={sandbox_url}")
    client = OpenAI(base_url=base_url, api_key=api_key)

    # Get actual model name if not specified
    if not model_name:
        try:
            model_name = client.models.list().data[0].id
        except Exception as e:
            print(f"Warning: Could not retrieve model name from API: {e}")
            model_name = model if model else "DeepSeek-V3-0324"
            print(f"Using fallback model name: {model_name}")
    print(f"DEBUG: model_name: {model_name}")

    # --- Sandbox Helper Functions & State ---
    MAX_SAME_ERROR = 2  # Max times to retry same failing code
    executed_snippets = {}  # {digest: (stdout, stderr, success)}
    already_seen_blocks = set()  # set of digests
    error_counter = {}  # {digest: count of consecutive failures}
    prev_code_digest = None  # Track previously executed block

    def _digest(code: str) -> str:
        """Create stable identifier for code snippets (ignoring whitespace)"""
        return hashlib.sha1(textwrap.dedent(code).strip().encode()).hexdigest()

    def prettify(code: str) -> str:
        """Format python code using black."""
        try:
            return format_str(code, mode=FileMode()).strip()
        except Exception as e:
            print(f"Warning: Black formatting failed: {e}. Using original code.")
            return code
    # --- End Sandbox Helpers ---

    # Make a copy of the original messages
    messages_copy = messages.copy()

    # Add system message for thinking process
    found_system = False
    for msg in messages_copy:
        if msg["role"] == "user" or msg["role"] == "system":
            found_system = True
            msg["content"] += "\nPlease think step by step, generating python code blocks when necessary. Your thinking may be interrupted, but you can continue. Code execution results will be provided."
            break
    if not found_system:
        messages_copy.insert(0, {
            "role": "system", # Prefer system role for instructions
            "content": "Please think step by step, generating python code blocks when necessary. Your thinking may be interrupted, but you can continue. Code execution results will be provided."
        })

    # PHASE 1: THINKING PHASE (with Sandbox Execution)
    thinking_responses = []
    remaining_tokens = max_tokens_thinking
    thinking_messages = messages_copy.copy()
    thinking_messages.append({"role": "assistant", "content": "Let me think step by step and execute code where needed:"})
    sandbox_executions = 0

    for thinking_round in range(100): # Max thinking rounds (arbitrary limit)
        if remaining_tokens <= 0:
            print("Thinking token budget exhausted.")
            break

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=thinking_messages,
                max_tokens=min(remaining_tokens, 8192), # Respect API limits per call
                temperature=0.8,
                top_p=0.8,
                stop=["</s>", "<|im_end|>"], # Stop may indicate code end
                timeout=timeout,
                extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}} if enable_thinking is not None else None,
            )

            thinking_content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            tokens_used = response.usage.completion_tokens
            remaining_tokens -= tokens_used

            thinking_responses.append(thinking_content)
            thinking_messages.append({"role": "assistant", "content": thinking_content})

            # --- Check for and Execute Code Block ---
            code_block = None
            current_digest = None
            result_message_for_model = None

            code_start_marker = "```python"
            code_start_index = thinking_content.rfind(code_start_marker)

            if code_start_index != -1 and sandbox_executions < max_sandbox_rounds:
                code_content_part = thinking_content[code_start_index + len(code_start_marker):]
                code_end_index = code_content_part.find("```")

                if code_end_index != -1:
                    code_block = code_content_part[:code_end_index].strip()
                else: # Likely stopped at end marker
                    code_block = code_content_part.strip()
                    if finish_reason == "stop" and thinking_content.endswith("```"): # Check if it really stopped at marker
                         pass # Already captured
                    elif finish_reason == "stop": # Stopped but maybe not exactly at marker
                         # Add closing marker back if needed for clarity in history, though model might regenerate it
                         thinking_messages[-1]["content"] += "<im_end>"


                if code_block:
                    current_digest = _digest(code_block)
                    print(f"Found code block (digest: {current_digest})")

                    # Check for repeats or previously seen blocks
                    if current_digest == prev_code_digest and thinking_round > 0:
                        print("Model repeated the same code block.")
                        result_message_for_model = "You've repeated the previous code block. Please try a different approach or continue thinking."
                    elif current_digest in already_seen_blocks:
                        print("Skipping already seen code block.")
                        result_message_for_model = "This code block was already attempted. Let's try a different approach or continue thinking."
                    else:
                        # Execute the code
                        already_seen_blocks.add(current_digest)
                        prev_code_digest = current_digest
                        sandbox_executions += 1
                        try:
                            print(f"Executing code snippet (digest: {current_digest})")
                            formatted_code = prettify(code_block)
                            res = requests.post(
                                sandbox_url,
                                json={"code": formatted_code, "language": "python"},
                                timeout=timeout,
                            )
                            res.raise_for_status()
                            res_json = res.json()

                            status_ok = res_json.get("status") == "Success"
                            run_res = res_json.get("run_result", {})
                            stdout = run_res.get("stdout", "")
                            stderr = run_res.get("stderr", "")
                            executed_snippets[current_digest] = (stdout, stderr, status_ok)

                            if status_ok:
                                error_counter[current_digest] = 0
                                result_text = stdout if stdout else "Execution finished with no stdout."
                            else:
                                if len(stderr) > 1000: stderr = stderr[-1000:] # Truncate long errors
                                error_counter[current_digest] = error_counter.get(current_digest, 0) + 1
                                result_text = f"--- Sandbox ERROR ---\n{stderr}"
                                if error_counter[current_digest] >= MAX_SAME_ERROR:
                                    result_text += f"\n\nThis exact error occurred {error_counter[current_digest]} times. Please try a different approach."

                        except Exception as e:
                            status_ok = False
                            error_counter[current_digest] = error_counter.get(current_digest, 0) + 1
                            executed_snippets[current_digest] = ("", str(e), False)
                            result_text = f"--- Sandbox Execution ERROR ---\n{e}"
                        
                        result_message_for_model = f"```output\n{result_text}\n```\nLet's continue based on this output."

                else: # Empty code block found
                    print("Extracted empty code block, asking model to continue.")
                    result_message_for_model = "You provided an empty code block. Please continue thinking or provide the correct code."
            elif sandbox_executions >= max_sandbox_rounds:
                 print(f"Max sandbox executions ({max_sandbox_rounds}) reached.")
                 result_message_for_model = "Maximum code executions reached for this thinking phase. Please continue based on the results so far or provide your final answer."


            # --- Feed back result or prompt to continue ---
            if result_message_for_model:
                # Feed back the sandbox result or error message
                thinking_messages.append({"role": "user", "content": result_message_for_model})
            elif finish_reason != "stop":
                # If the model didn't stop naturally (e.g., length limit), prompt it to continue
                 thinking_messages.append({"role": "user", "content": "Please continue thinking step by step."})
            else:
                 # add Wait, and prompt the model to continue
                 thinking_messages.append({"role": "user", "content": "Wait, let's verify our previous answer."})


        except Exception as e:
            print(f"Error during thinking/sandbox phase: {e}")
            thinking_responses.append(f"Error occurred during thinking: {str(e)}")
            # Add error to history so model is aware
            thinking_messages.append({"role": "user", "content": f"An error occurred: {e}. Please proceed to your final answer based on the thinking so far."})
            break # Exit thinking loop on error

    # PHASE 2: FINAL ANSWER PHASE
    combined_thinking = " ".join(thinking_responses) # Simple combination for now
    final_messages = messages_copy.copy() # Start from original prompt + system message
    final_messages.append({
        "role": "assistant",
        "content": combined_thinking + "\n\nBased on the step-by-step thinking and code execution results above, here is the FINAL ANSWER:"
    }) # Use the full thinking history including sandbox results

    try:
        print("Generating final answer...")
        final_response = client.chat.completions.create(
            model=model_name,
            messages=final_messages, # Provide the thinking context
            max_tokens=max_output_tokens,
            temperature=0.8, # Can adjust temperature for final answer
            top_p=0.8,
            stop=["</s>", "<|im_end|>"], # Standard stops for final answer
            timeout=timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}} if enable_thinking is not None else None,
        )
        final_answer = final_response.choices[0].message.content or ""
    except Exception as e:
        print(f"Error during final answer phase: {e}")
        final_answer = f"Error occurred during final answer generation: {str(e)}"

    # Construct full response including thinking and final answer
    # Use the thinking_messages history for a more accurate full response
    full_response_messages = thinking_messages + [{"role": "assistant", "content": final_answer}]


    # Return format consistent with budget forcing, but add full message history
    return {
        "content": final_answer,
        "full_response": full_response_messages # Return the list of messages
    }
# --- End: New function ---

@retry(
    stop=stop_after_attempt(50),
    wait=wait_exponential(multiplier=1, min=1, max=1000),
    retry=retry_if_exception_type(OpenAIError),
)
def request_with_thinking_control(messages, timeout=2000, max_tokens=8192, base_url="", api_key="", model="", model_name=None, enable_thinking=False):
    """
    Standard chat completion request with thinking control support.
    This function supports the enable_thinking parameter for simple modes.
    """
    client = OpenAI(base_url=base_url, api_key=api_key)
    if not model_name:
        try:
            model_name = client.models.list().data[0].id
        except Exception as e:
            print(f"Warning: Could not retrieve model name from API: {e}")
            model_name = model if model else "DeepSeek-V3-0324"
            print(f"Using fallback model name: {model_name}")
    else:
        model_name = model_name
    print(f"DEBUG: model_name: {model_name}")
    print(f"DEBUG: messages: {messages}")
    print(f"DEBUG: enable_thinking: {enable_thinking}")
    
    extra_body = None
    if enable_thinking is not None:
        extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.8,
        top_p=0.8,
        stop=["</s>", "<|im_end|>"], # Standard stops for final answer
        max_tokens=20000,
        timeout=timeout,
        extra_body=extra_body,
    )
    return response

@retry(
    stop=stop_after_attempt(50),
    wait=wait_exponential(multiplier=1, min=1, max=1000),
    retry=retry_if_exception_type(OpenAIError),
)
def request_with_agent(
        messages: list[dict],
        tools: list[dict],
        *,
        sandbox_url="http://localhost:8080/run_code",
        max_rounds: int = 10,
        max_tokens: int = 30000,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        model_name: str = None,
        timeout: int = 3600,
        enable_thinking=True,
    ) -> list[dict]:
    """
    Generic agent loop that lets a vLLM‑served model pick and call tools.
    Requires vLLM to be launched with:
        vllm serve ... --enable-auto-tool-choice --tool-call-parser hermes
    """
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    # Get model name if not provided
    if not model_name:
        try:
            model_name = client.models.list().data[0].id
        except Exception as e:
            print(f"Warning: Could not retrieve model name from API: {e}")
            model_name = model if model else "DeepSeek-V3-0324"
            print(f"Using fallback model name: {model_name}")
    print(f"DEBUG: model_name: {model_name}")
    full_messages = list(messages)
    # clean_messages 发给模型用（已 strip 掉 internal thoughts）
    clean_messages = sanitize_messages(messages)
    print(f"Starting agent loop with max_rounds={max_rounds}, sandbox_url={sandbox_url}")
    THINK_RE = re.compile(r"<think>.*?</think>|<think>", flags=re.DOTALL)
    # Track consecutive thinking-only responses
    consecutive_thinking_only = 0
    max_thinking_only_retries = 3
    
    def prune_verbal_messages(msgs):
        """
        Return a new list where we drop any assistant messages that:
        - have no tool_calls, AND
        - whose content (after removing all <think> tags) is empty or whitespace-only.
        """
        pruned = []
        for m in msgs:
            if m["role"] == "assistant":
                raw = m.get("content", "")
                stripped = THINK_RE.sub("", raw).strip()
                if m.get("tool_calls") or stripped:
                    pruned.append(m)
                # else: drop it
            else:
                pruned.append(m)
        return pruned
    
    for round_num in range(max_rounds):
        print(f"▶ AGENT ROUND {round_num + 1}")
        
        try:
            # Calculate approximate prompt length from all messages
            prompt_length = sum(len(msg["content"]) if msg.get("content") else 0 for msg in clean_messages)
            response = None
            if enable_thinking == False:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=clean_messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=max_tokens,
                    temperature=0.8,
                    timeout=timeout,
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=clean_messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=max_tokens,
                    temperature=0.8,
                    timeout=timeout,
                )
            
            msg = response.choices[0].message
            full_messages.append(msg.model_dump())  # keep raw assistant node
            sanitized_reply = msg.model_dump().copy()
            sanitized_reply.pop("reasoning_content", None)
            sanitized_reply["content"] = THINK_RE.sub("", sanitized_reply.get("content","")).strip()
            print(f"Model response: {msg.content[:100] if msg.content else 'No content'}...")
            
            # Check if response content is exactly "<think>"
            is_exact_think_only = (msg.content and msg.content.strip() == "<think>" and not msg.tool_calls)
            
            if is_exact_think_only:
                consecutive_thinking_only += 1
                print(f"Detected exact '<think>' response ({consecutive_thinking_only}/{max_thinking_only_retries})")
                clean_messages = prune_verbal_messages(clean_messages)
                if consecutive_thinking_only >= max_thinking_only_retries:
                    # Create filtered messages for fallback: keep user prompts, tool messages, and assistant messages with tool calls
                    try:
                        fallback_response = client.chat.completions.create(
                            model=model_name,
                            messages=clean_messages,
                            max_tokens=max_tokens,
                            temperature=0.8,
                            timeout=timeout,
                        )
                        fallback_msg = fallback_response.choices[0].message
                        full_messages.append(fallback_msg.model_dump())
                        print(f"Fallback response: {fallback_msg.content[:100] if fallback_msg.content else 'No content'}...")
                        break  # Exit the agent loop
                    except Exception as e:
                        print(f"Error during fallback request: {e}")
                        full_messages.append({
                            "role": "assistant",
                            "content": f"Error during fallback execution: {e}"
                        })
                        break
                else:
                    # Remove the thinking-only response from full_messages
                    full_messages.pop()
                    
                    print("Retrying with filtered conversation (keeping only prompts, tool calls, and tool responses)...")
                    continue
            else:
                # Reset consecutive thinking counter if we get a valid response
                consecutive_thinking_only = 0
                clean_messages.append(sanitized_reply)
            
            # 1️⃣ Did the model decide to call a tool?
            if not msg.tool_calls:
                # Check if the content has a code block
                code_block_pattern = re.compile(r'```python\n(.*?)\n```', re.DOTALL)
                content = msg.content if msg.content else ""
                code_match = code_block_pattern.search(content)
                if code_match:
                    # Extract the code
                    code = code_match.group(1).strip()
                    print(f"Found a code block in the response. Extracted code: {code[:100]}...")
                    # Create a virtual tool call
                    virtual_tool_call_id = f"virtual_{round_num}_{len(full_messages)}"
                    # We'll simulate a tool call for run_python
                    fn_name = "run_python"
                    # Execute the code in the sandbox
                    try:
                        res = requests.post(
                            sandbox_url,
                            json={"code": code,
                                  "timeout_sec": 300,  # default timeout
                                  "language": "python"},
                            timeout=180,
                        )
                        res.raise_for_status()
                        tool_response = json.dumps(res.json(), ensure_ascii=False)
                        print(f"Code execution result: {tool_response[:100]}...")
                    except Exception as e:
                        print(f"Code execution error: {e}")
                        tool_response = json.dumps(
                            {"status": "Error", "run_result": {}, 
                             "stderr": f"Sandbox execution error: {e}"})
                    
                    # Append the tool response
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": virtual_tool_call_id,
                        "content": tool_response,
                    })
                    clean_messages.append({
                        "role": "tool",
                        "tool_call_id": virtual_tool_call_id,
                        "content": tool_response,
                    })
                    
                    # Then add the user message to prompt the model to continue
                    full_messages.append({
                        "role": "user",
                        "content": "Based on the results from the tool calls, please continue your reasoning and tool calling. If you’ve reached a solution, present the final answer in format FINAL ANSWER: [[your answer]]",
                    })
                    clean_messages.append({
                        "role": "user",
                        "content": "Based on the results from the tool calls, please continue your reasoning and tool calling. If you’ve reached a solution, present the final answer in format FINAL ANSWER: [[your answer]]",
                    })
                    
                    # We do not break, so the loop will continue
                    print("Virtual tool call processed. Continuing agent loop.")
                else:
                    print("No tool calls and no code block detected, ending agent loop")
                    break  # normal answer → exit
            else:
                print(f"Found {len(msg.tool_calls)} tool calls")
                
                # 2️⃣ Execute every tool call that came back
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)
                    call_id = tc.id
                    
                    print(f"Executing tool: {fn_name} with args: {fn_args}")
                    
                    if fn_name == "run_python":  # ✨ only tool we expose
                        try:
                            res = requests.post(
                                sandbox_url,
                                json={"code": fn_args["code"],
                                      "timeout_sec": fn_args.get("timeout_sec", 300),
                                      "language": "python"},
                                timeout=180,
                            )
                            res.raise_for_status()
                            tool_response = json.dumps(res.json(), ensure_ascii=False)
                            print(f"Tool execution result: {tool_response[:100]}...")
                        except Exception as e:
                            print(f"Tool execution error: {e}")
                            tool_response = json.dumps(
                                {"status": "Error", "run_result": {}, 
                                 "stderr": f"Sandbox execution error: {e}"})
                    else:
                        tool_response = json.dumps(
                            {"status": "Error", "run_result": {}, 
                             "stderr": f"Unknown tool {fn_name}"})
                    
                    # 3️⃣ Feed the tool result back so the model can continue
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": tool_response,
                    })
                    clean_messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": tool_response,
                    })
                
                # After processing all tool calls, add a message to prompt the model to continue.
                full_messages.append({
                    "role": "user",
                    "content": "Based on the results from the tool calls, please continue your reasoning and tool calling. If you’ve reached a solution, present the final answer in format FINAL ANSWER: [[your answer]]",
                })
                clean_messages.append({
                    "role": "user",
                    "content": "Based on the results from the tool calls, please continue your reasoning and tool calling. If you’ve reached a solution, present the final answer in format FINAL ANSWER: [[your answer]]",
                })

        except Exception as e:
            print(f"Error during agent round {round_num + 1}: {e}")
            # Add error message to conversation
            full_messages.append({
                "role": "assistant",
                "content": f"Error occurred during agent execution: {e}"
            })
            clean_messages.append({
                "role": "assistant",
                "content": f"Error occurred during agent execution: {e}"
            })
            break
    
    print(f"Agent loop completed after {round_num + 1} rounds")
    return full_messages

def sanitize_messages(messages):
    """
    返回一个新的消息列表：
      - 去掉 reasoning_content
      - 删除 content 中所有 <think>…</think> 区块
    """
    clean = []
    THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
    for msg in messages:
        # 深拷贝一份，防止篡改原消息
        m = msg.copy()
        # 丢弃内部推理
        m.pop("reasoning_content", None)
        # 清理 <think>…</think>
        m["content"] = THINK_RE.sub("", m.get("content", "")).strip()
        clean.append(m)
    return clean

def infer(prompts, historys, **kwargs):
    print(f"kwargs: {kwargs}")
    model = kwargs.get('model')
    base_url = kwargs.get('base_url')
    api_key = kwargs.get('api_key')
    model_name = kwargs.get('model_name', None)
    call_type = kwargs.get('call_type', 'api_chat')
    code_mode = kwargs.get('code_mode', 'noncode')
    print(f"DEBUG: code_mode: {code_mode}")
    
    # Budget forcing parameters
    use_budget_forcing = kwargs.get('use_budget_forcing', False)
    max_tokens_thinking = kwargs.get('max_tokens_thinking', 32000)
    max_output_tokens = kwargs.get('max_output_tokens', 8192)
    
    use_sandbox = False
    if code_mode in ['sandbox', 'sandbox_nothink']:
        use_sandbox = True
    else:
        use_sandbox = False

    # Only pass enable_thinking=False when _nothink is in code_mode
    should_disable_thinking = code_mode.endswith('_nothink')

    try:
        if call_type == 'api_chat':
            if isinstance(prompts, list):
                if len(prompts) > 1:
                    print(f'[Warning] infer/models/openai_api.py: Multiple prompts detected, only the first one will be processed')
                prompts = prompts[0]
                historys = historys[0]
            
            # Build the conversation from prompts and history
            messages = build_conversation(historys, prompts)
            
            if use_budget_forcing and not use_sandbox:
                # Use budget forcing approach
                print("Using budget forcing for API request")
                response_obj = request_with_budget_forcing(
                    messages,
                    max_tokens_thinking=max_tokens_thinking,
                    max_output_tokens=max_output_tokens,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    model_name=model_name,
                )
                # Return both the final answer and the full response
                return [response_obj]
            elif use_sandbox and not use_budget_forcing:
                # Use sandbox approach
                kwargs_sandbox = {
                    'messages': messages,
                    'max_tokens': config_wrapper.max_tokens,
                    'base_url': base_url,
                    'api_key': api_key,
                    'model': model,
                    'model_name': model_name,
                }
                if should_disable_thinking:
                    kwargs_sandbox['enable_thinking'] = False
                final_messages = request_with_sandbox(**kwargs_sandbox)
                response = final_messages[-1]["content"]   # assistant's last message
                # Store the complete conversation in a special format that doesn't break existing code
                return [{"content": response, "full_conversation": final_messages}]
            elif use_budget_forcing and use_sandbox:
                kwargs_budget_sandbox = {
                    'messages': messages,
                    'max_tokens_thinking': max_tokens_thinking,
                    'max_output_tokens': max_output_tokens,
                    'base_url': base_url,
                    'api_key': api_key,
                }
                if should_disable_thinking:
                    kwargs_budget_sandbox['enable_thinking'] = False
                response_obj = request_with_budget_forcing_and_sandbox(**kwargs_budget_sandbox)
                return [response_obj]
            elif code_mode in ["agent", "agent_nothink"]:
                # Use agent approach with tool calling
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "run_python",
                            "description": "Execute Python code and return stdout/stderr",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "code": {"type": "string", "description": "Python code to execute"},
                                    "timeout_sec": {"type": "integer", "default": 300, "description": "Timeout in seconds"}
                                },
                                "required": ["code"]
                            }
                        }
                    }
                ]
                
                kwargs_agent = {
                    'messages': messages,
                    'tools': tools,
                    'sandbox_url': kwargs.get('sandbox_url', 'http://localhost:8080/run_code'),
                    'max_rounds': kwargs.get('max_rounds', 6),
                    'max_tokens': config_wrapper.max_tokens,
                    'base_url': base_url,
                    'api_key': api_key,
                    'model': model,
                    'model_name': model_name,
                }
                if should_disable_thinking:
                    kwargs_agent['enable_thinking'] = False
                full_convo = request_with_agent(**kwargs_agent)
                
                # Extract the final response
                response = full_convo[-1]["content"] if full_convo[-1]["role"] == "assistant" else ""
                return [{"content": response, "full_conversation": full_convo}]
            elif code_mode in ['pot_nothink', 'noncode_nothink']:
                # Handle simple modes with thinking control
                kwargs_thinking = {
                    'messages': messages,
                    'max_tokens': config_wrapper.max_tokens,
                    'base_url': base_url,
                    'api_key': api_key,
                    'model': model,
                    'model_name': model_name,
                }
                if should_disable_thinking:
                    kwargs_thinking['enable_thinking'] = False
                messages_response = request_with_thinking_control(**kwargs_thinking)
                response = messages_response.choices[0].message.content
                try:
                    reasoning_content = ""
                    if hasattr(messages_response.choices[0].message, "reasoning_content") and not should_disable_thinking:
                        reasoning_content = messages_response.choices[0].message.reasoning_content
                        response = reasoning_content + "\n\n" + response
                except:
                    print(f"DEBUG: No reasoning content found for the response: {response}")
                return [response]
            else:
                # Standard chat API request
                messages = request(messages, max_tokens=config_wrapper.max_tokens, base_url=base_url, api_key=api_key, model=model, model_name=model_name)
                response = messages.choices[0].message.content
                try:
                    reasoning_content = ""
                    if hasattr(messages.choices[0].message, "reasoning_content") and not should_disable_thinking:
                        reasoning_content = messages.choices[0].message.reasoning_content
                        response = reasoning_content + "\n\n" + response
                except:
                    print(f"DEBUG: No reasoning content found for the response: {response}")
                return [response]
        elif call_type == 'api_base':
            # Base model API request
            response = request_to_base_model(prompts, max_tokens=config_wrapper.max_tokens, base_url=base_url, api_key=api_key, model=model, model_name=model_name).choices[0].text
            return [response]
    except Exception as e:
        response = {"error": str(e)}
    # print(response)
    return [response]

    

