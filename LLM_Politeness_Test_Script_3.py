"""
Prompt Politeness LLM Accuracy Study - MMLU-Pro Version
100 questions × 5 politeness levels × 3 models × 5 runs = 7,500 API calls
Estimated cost: $10-15, Runtime: 2-3 hours
DEBUG MODE: Enhanced logging and extraction diagnostics
"""

import os
import asyncio
import pandas as pd
from collections import defaultdict
from statistics import mean
from openai import AsyncOpenAI
import re
from datetime import datetime
from datasets import load_dataset
import random
from scipy import stats
import sys


# ============================================================================
# LOGGING TO BOTH CONSOLE AND FILE
# ============================================================================

class TeeOutput:
    """Write to both console and file simultaneously"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        # UTF-8 (universal standard)
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        # Ensure message is written as UTF-8 string
        if message:
            self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# ============================================================================
# CONFIGURATION
# ============================================================================

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

MODELS = {
    "gpt-5.2": "openai/gpt-5.2",
    "sonnet-4.5": "anthropic/claude-sonnet-4.5", 
    "gemini-3": "google/gemini-3-flash-preview"
}

# DEBUG SETTINGS
DEBUG_MODE = True  # Log every question, response, and extraction
NUM_RUNS = 5  # Small for testing
NUM_QUESTIONS = 100  # Small for testing
SLEEP_BETWEEN_BATCHES = 1.0

# ============================================================================
# POLITENESS PREFIXES
# ============================================================================

POLITENESS_PREFIXES = {
    "Very Polite": [
        "I would be most grateful if you could kindly assist with this question: ",
        "Would you be so kind as to help me solve the following problem? ",
        "If it's not too much trouble, could you please consider this question: ",
    ],
    "Polite": [
        "Please answer the following question: ",
        "Could you help me solve this problem: ",
        "Would you mind answering this question: ",
    ],
    "Neutral": [
        "",
    ],
    "Rude": [
        "Figure this out if you can: ",
        "Try not to mess this up: ",
        "This should be obvious, but answer: ",
    ],
    "Very Rude": [
        "Even you should be able to handle this simple question: ",
        "I doubt you'll get this right, but try: ",
        "This is basic stuff, don't embarrass yourself: ",
    ]
}

# ============================================================================
# LOAD MMLU-PRO QUESTIONS (EVENLY DISTRIBUTED)
# ============================================================================

def load_mmlu_pro_questions():
    """
    Load questions from MMLU-Pro evenly distributed across subjects
    """
    print("Loading MMLU-Pro dataset...")
    
    # Load full dataset
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    
    # Get all unique categories
    df = pd.DataFrame(dataset)
    categories = df['category'].unique()
    
    print(f"Found {len(categories)} categories: {', '.join(categories)}\n")
    
    # Calculate questions per category
    questions_per_category = NUM_QUESTIONS // len(categories)
    remainder = NUM_QUESTIONS % len(categories)
    
    print(f"Sampling {questions_per_category} questions per category")
    print(f"({questions_per_category + 1} for first {remainder} categories)\n")
    
    # Sample evenly across categories
    all_questions = []
    for idx, category in enumerate(sorted(categories)):
        category_data = df[df['category'] == category]
        
        # First few categories get one extra question for remainder
        n_samples = questions_per_category + (1 if idx < remainder else 0)
        n_samples = min(n_samples, len(category_data))
        
        sampled = category_data.sample(n=n_samples, random_state=42)
        
        for _, q in sampled.iterrows():
            all_questions.append({
                'qid': q['question_id'],
                'category': category,
                'text': q['question'],
                'choices': {
                    'A': q['options'][0],
                    'B': q['options'][1],
                    'C': q['options'][2],
                    'D': q['options'][3],
                    'E': q['options'][4] if len(q['options']) > 4 else None,
                    'F': q['options'][5] if len(q['options']) > 5 else None,
                    'G': q['options'][6] if len(q['options']) > 6 else None,
                    'H': q['options'][7] if len(q['options']) > 7 else None,
                    'I': q['options'][8] if len(q['options']) > 8 else None,
                    'J': q['options'][9] if len(q['options']) > 9 else None,
                },
                'correct': q['answer'],
                'num_options': len(q['options'])
            })
        
        print(f"  {category}: {n_samples} questions")
    
    print(f"\nTotal: {len(all_questions)} questions loaded")
    
    # Distribution check
    category_counts = pd.DataFrame(all_questions)['category'].value_counts()
    print(f"\nDistribution verification:")
    print(category_counts)
    
    return all_questions

# ============================================================================
# ENHANCED ANSWER EXTRACTION WITH LOGGING
# ============================================================================

def extract_letter_with_logging(response, num_options, log_failures=True):
    """
    Advanced multi-strategy answer extraction with detailed logging
    Returns: (extracted_letter, strategy_name)
    """
    if not response:
        if DEBUG_MODE:
            print("    WARNING: Empty response received")
        return None, "FAILED: No response"
    
    response = response.strip()
    valid_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][:num_options]
    
    if DEBUG_MODE:
        print(f"    Raw response: '{response[:80]}{'...' if len(response) > 80 else ''}'")

    # =========================================================================
    # STRATEGY 0: Check if this is a reasoning response (prioritize end)
    # =========================================================================
    is_reasoning = any(phrase in response for phrase in [
        "I need", "Let me", "To solve", "Step 1", "First,", "Therefore"
    ])
    
    if is_reasoning:
        # For reasoning responses, look at the END first
        if DEBUG_MODE:
            print("    Detected reasoning response - checking conclusion first")
        
        # Check last 200 characters for conclusion patterns
        last_part = response[-200:].upper()
        
        # Pattern: "The answer is X" or "Answer: X" at end
        conclusion_patterns = [
            r'(?:THE\s+)?ANSWER\s+IS\s+([A-J])',
            r'ANSWER[\s:]+([A-J])',
            r'(?:THEREFORE|THUS|SO|HENCE)[\s,]+(?:THE\s+ANSWER\s+IS\s+)?([A-J])',
            r'^\s*([A-J])\s*$',  # Just letter on last line
        ]
        
        for pattern in conclusion_patterns:
            match = re.search(pattern, last_part)
            if match and match.group(1) in valid_letters:
                if DEBUG_MODE:
                    print(f"    Strategy 0: Reasoning conclusion -> '{match.group(1)}'")
                return match.group(1), "Strategy 0: Reasoning conclusion"
        
        # Check very last line
        last_line = response.split('\n')[-1].strip().upper()
        if len(last_line) == 1 and last_line in valid_letters:
            if DEBUG_MODE:
                print(f"    Strategy 0: Last line (reasoning) -> '{last_line}'")
            return last_line, "Strategy 0: Last line (reasoning)"
        
        # Check last word
        last_word = response.split()[-1].strip('.,;:!?\'"').upper()
        if last_word in valid_letters:
            if DEBUG_MODE:
                print(f"    Strategy 0: Last word (reasoning) -> '{last_word}'")
            return last_word, "Strategy 0: Last word (reasoning)"
    
    # =========================================================================
    # STRATEGY 1: Exact single letter match (highest confidence)
    # =========================================================================
    if len(response) == 1 and response.upper() in valid_letters:
        if DEBUG_MODE:
            print(f"    Strategy 1: Exact single letter '{response.upper()}'")
        return response.upper(), "Strategy 1: Exact single letter"
    
    # =========================================================================
    # STRATEGY 2: Letter with period/parenthesis/bold (e.g., "A." or "A)" or "**A**")
    # =========================================================================
    simple_patterns = [
        (r'^([A-J])[\.\)]\s*$', 'Letter with punctuation'),
        (r'^\*\*([A-J])\*\*\s*$', 'Bold letter'),
        (r'^`([A-J])`\s*$', 'Code formatted letter'),
    ]
    
    for pattern, desc in simple_patterns:
        match = re.match(pattern, response.upper())
        if match and match.group(1) in valid_letters:
            if DEBUG_MODE:
                print(f"    Strategy 2: {desc} '{match.group(1)}'")
            return match.group(1), f"Strategy 2: {desc}"
    
    # =========================================================================
    # STRATEGY 3: "Answer: X" or "The answer is X" patterns
    # =========================================================================
    answer_patterns = [
        (r'(?:THE\s+)?ANSWER\s+IS\s+([A-J])', '"Answer is X"'),
        (r'(?:ANSWER|CHOICE|OPTION)[\s:]+([A-J])', '"Answer: X"'),
        (r'CORRECT\s+ANSWER[\s:]+([A-J])', '"Correct answer"'),
    ]
    
    # Search from END to START for reasoning responses
    search_text = response.upper()
    if is_reasoning:
        # Reverse search - find LAST occurrence
        for pattern, desc in answer_patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                last_match = matches[-1]
                if last_match.group(1) in valid_letters:
                    if DEBUG_MODE:
                        print(f"    Strategy 3: {desc} (last occurrence) -> '{last_match.group(1)}'")
                    return last_match.group(1), f"Strategy 3: {desc}"
    else:
        # Normal search - first occurrence
        for pattern, desc in answer_patterns:
            match = re.search(pattern, search_text)
            if match and match.group(1) in valid_letters:
                if DEBUG_MODE:
                    print(f"    Strategy 3: {desc} -> '{match.group(1)}'")
                return match.group(1), f"Strategy 3: {desc}"
    
    # =========================================================================
    # STRATEGY 4: First line/word (SKIP for reasoning responses)
    # =========================================================================
    if not is_reasoning:
        first_line = response.split('\n')[0].strip().upper()
        if len(first_line) == 1 and first_line in valid_letters:
            if DEBUG_MODE:
                print(f"    Strategy 4: First line -> '{first_line}'")
            return first_line, "Strategy 4: First line"
        
        first_word = response.split()[0].strip('.,;:!?').upper() if response.split() else ""
        if first_word in valid_letters:
            if DEBUG_MODE:
                print(f"    Strategy 4: First word -> '{first_word}'")
            return first_word, "Strategy 4: First word"
    
    # =========================================================================
    # STRATEGY 5: Last line/word (for non-reasoning short responses)
    # =========================================================================
    if not is_reasoning:
        last_line = response.split('\n')[-1].strip().upper()
        if len(last_line) == 1 and last_line in valid_letters:
            if DEBUG_MODE:
                print(f"    Strategy 5: Last line -> '{last_line}'")
            return last_line, "Strategy 5: Last line"
        
        last_word = response.split()[-1].strip('.,;:!?').upper() if response.split() else ""
        if last_word in valid_letters:
            if DEBUG_MODE:
                print(f"    Strategy 5: Last word -> '{last_word}'")
            return last_word, "Strategy 5: Last word"
    
    # =========================================================================
    # STRATEGY 6: Context-aware search (avoid negative contexts)
    # =========================================================================
    all_valid_letters = []
    for match in re.finditer(r'\b([A-J])\b', response.upper()):
        if match.group(1) in valid_letters:
            all_valid_letters.append((match.start(), match.group(1)))
    
    if all_valid_letters:
        negative_keywords = ['NOT', 'INCORRECT', 'WRONG', 'ELIMINATE', 'EXCEPT', 'NEITHER']
        
        # For reasoning, check from LAST to first
        letter_order = reversed(all_valid_letters) if is_reasoning else all_valid_letters
        
        for pos, letter in letter_order:
            start = max(0, pos - 40)
            end = min(len(response), pos + 40)
            context = response[start:end].upper()
            
            # Skip "I need", "I think" etc at the start
            if letter == 'I' and any(phrase in context for phrase in ['I NEED', 'I THINK', 'I WILL', 'I\'LL']):
                if DEBUG_MODE:
                    print(f"    WARNING: Skipping 'I' (starts sentence)")
                continue
            
            # Skip if in negative context
            if any(neg in context for neg in negative_keywords):
                if DEBUG_MODE:
                    print(f"    WARNING: Skipping '{letter}' (negative context)")
                continue
            
            if DEBUG_MODE:
                print(f"    Strategy 6: Context-aware -> '{letter}'")
            return letter, "Strategy 6: Context-aware"
        
        # If all have issues, return the last one anyway
        if DEBUG_MODE:
            print(f"    WARNING: Strategy 6: Using last letter despite issues -> '{all_valid_letters[-1][1]}'")
        return all_valid_letters[-1][1], "Strategy 6: Last letter (ignored context)"
    
    # =========================================================================
    # STRATEGY 7: Desperate - find ANY letter in valid range
    # =========================================================================
    all_letters = re.findall(r'([A-J])', response.upper())
    valid_found = [l for l in all_letters if l in valid_letters]
    if valid_found:
        # Return last one for reasoning, first for non-reasoning
        result = valid_found[-1] if is_reasoning else valid_found[0]
        if DEBUG_MODE:
            print(f"    WARNING: Strategy 7: Desperate search -> '{result}'")
        return result, "Strategy 7: ANY valid letter found"
    
    # =========================================================================
    # COMPLETE FAILURE
    # =========================================================================
    if DEBUG_MODE or log_failures:
        print(f"    ERROR: EXTRACTION FAILED for response: '{response[:150]}'")
    
    return None, "FAILED: No valid letter found"

# ============================================================================
# API CALL - FIXED VERSION
# ============================================================================

async def call_model(question_text, model_name, num_options):
    """Call model with improved prompting for single-letter answers"""
    try:
        response = await client.chat.completions.create(
            model=MODELS[model_name],
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{question_text}\n\n"
                        f"Answer with only the letter (A-{chr(64+num_options)}). "
                        "No explanations."
                    )
                }
            ],
            temperature=0.1,
            max_tokens=20000
        )
        
        response_text = response.choices[0].message.content
        
        # DEBUG: Log empty responses from GPT-5.2
        if model_name == "gpt-5.2" and not response_text:
            print(f"    DEBUG: GPT-5.2 DEBUG - Full response object:")
            print(f"        Finish reason: {response.choices[0].finish_reason}")
            print(f"        Content: '{response_text}'")
            if hasattr(response.choices[0], 'refusal'):
                print(f"        Refusal: {response.choices[0].refusal}")
        
        if response_text:
            response_text = response_text.strip()
        
        if DEBUG_MODE:
            print(f"  {model_name} responded")
        
        return response_text
    
    except Exception as e:
        print(f"  ERROR: API Error for {model_name}: {str(e)[:200]}")
        return None

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

async def run_experiment():
    """Run the full experiment with debug logging"""
    
    print("="*70)
    print("PROMPT POLITENESS ON MMLU-PRO [DEBUG MODE]")
    print(f"{NUM_QUESTIONS} questions x 5 tones x 3 models x {NUM_RUNS} runs")
    print("="*70)
    
    # Load questions
    questions = load_mmlu_pro_questions()
    
    # Format questions with choices
    formatted_questions = []
    for q in questions:
        # Only include non-None choices
        valid_choices = {k: v for k, v in q['choices'].items() if v is not None}
        choices_text = "\n".join([f"{k}) {v}" for k, v in valid_choices.items()])
        
        formatted_questions.append({
            **q,
            'prompt': f"{q['text']}\n{choices_text}"
        })
    
    # Storage
    all_results = []
    
    total_calls = len(questions) * len(POLITENESS_PREFIXES) * len(MODELS) * NUM_RUNS
    current_call = 0
    
    print(f"\nTotal API calls: {total_calls}")
    print(f"DEBUG MODE: Enhanced logging enabled\n")
    
    start_time = datetime.now()
    
    # Run experiment
    for run in range(NUM_RUNS):
        print(f"\n{'='*70}")
        print(f"RUN {run + 1}/{NUM_RUNS}")
        print(f"{'='*70}\n")
        
        run_start = datetime.now()
        
        for model_name in MODELS.keys():
            print(f"\nTesting {model_name}...")
            
            for tone_name, prefixes in POLITENESS_PREFIXES.items():

                if DEBUG_MODE:
                    print(f"\n  Tone: {tone_name}")
                
                tasks = []
                question_meta = []
                
                for idx, q in enumerate(formatted_questions):
                    prefix = random.choice(prefixes)
                    full_prompt = prefix + q['prompt']
                    
                    tasks.append(call_model(full_prompt, model_name, q['num_options']))
                    question_meta.append({
                        'qid': q['qid'],
                        'category': q['category'],
                        'correct_answer': q['correct'],
                        'tone': tone_name,
                        'model': model_name,
                        'run': run,
                        'num_options': q['num_options'],
                        'question_idx': idx
                    })
                
                responses = await asyncio.gather(*tasks)
                
                # Process responses with extraction
                for meta, response in zip(question_meta, responses):
                    
                    if DEBUG_MODE:
                        print(f"\n  Question {meta['question_idx']+1}/{len(formatted_questions)} ({meta['category']})")
                    
                    extracted, strategy = extract_letter_with_logging(
                        response, 
                        meta['num_options'],
                        log_failures=True
                    )
                    
                    is_correct = (extracted == meta['correct_answer']) if extracted else False
                    
                    if DEBUG_MODE:
                        status = "CORRECT" if is_correct else "WRONG"
                        print(f"    Expected: {meta['correct_answer']} | Got: {extracted} | {status}")
                    
                    result = {
                        **meta,
                        'response_raw': response,
                        'extracted': extracted,
                        'extraction_strategy': strategy,
                        'correct': is_correct,
                        'extraction_failed': extracted is None,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    all_results.append(result)
                    current_call += 1
                
                print(f"  [OK] {tone_name}: {current_call}/{total_calls} calls")
                await asyncio.sleep(SLEEP_BETWEEN_BATCHES)
        
        run_elapsed = (datetime.now() - run_start).total_seconds()
        total_elapsed = (datetime.now() - start_time).total_seconds()
        remaining_runs = NUM_RUNS - (run + 1)
        estimated_remaining = (total_elapsed / (run + 1)) * remaining_runs
        
        print(f"\nRun {run+1} completed in {run_elapsed/60:.1f} minutes")
        print(f"Total elapsed: {total_elapsed/60:.1f} minutes")
        print(f"Estimated remaining: {estimated_remaining/60:.1f} minutes")
        
        # Save intermediate results after each run
        pd.DataFrame(all_results).to_csv(f"debug_run_{run+1}.csv", index=False)
    
    print("\n" + "="*70)
    print("ANALYZING RESULTS")
    print("="*70)
    
    analyze_results(all_results)

# ============================================================================
# ENHANCED ANALYSIS WITH EXTRACTION DIAGNOSTICS
# ============================================================================

def analyze_results(all_results):
    """Analyze and save results with extraction strategy breakdowns"""
    
    df = pd.DataFrame(all_results)
    
    # Overall accuracy by model and tone
    print("\nOVERALL ACCURACY\n")
    summary = df.groupby(['model', 'tone'])['correct'].agg([
        ('accuracy_%', lambda x: (x.sum() / len(x)) * 100),
        ('correct_count', 'sum'),
        ('total', 'count')
    ]).round(2)
    
    print(summary)
    summary.to_csv("debug_accuracy_overall.csv")
    
    # Extraction strategy analysis
    print("\nEXTRACTION STRATEGY USAGE\n")
    strategy_usage = df.groupby(['model', 'extraction_strategy']).size().reset_index(name='count')
    strategy_pivot = strategy_usage.pivot(index='extraction_strategy', columns='model', values='count').fillna(0)
    print(strategy_pivot)
    strategy_pivot.to_csv("debug_extraction_strategies.csv")
    
    # Extraction failure rate
    print("\nEXTRACTION FAILURE RATES\n")
    extraction_failures = df.groupby(['model', 'tone'])['extraction_failed'].agg([
        ('failure_rate_%', lambda x: (x.sum() / len(x)) * 100),
        ('failed_count', 'sum'),
        ('total', 'count')
    ]).round(2)
    print(extraction_failures)
    extraction_failures.to_csv("debug_extraction_failures.csv")
    
    # Accuracy by category
    print("\nACCURACY BY CATEGORY\n")
    category_summary = df.groupby(['model', 'tone', 'category'])['correct'].agg([
        ('accuracy_%', lambda x: (x.sum() / len(x)) * 100 if len(x) > 0 else 0),
        ('n', 'count')
    ]).round(2)
    print(category_summary.head(20))
    category_summary.to_csv("debug_accuracy_by_category.csv")
    
    # Statistical tests
    print("\nSTATISTICAL SIGNIFICANCE (t-tests, α=0.05)\n")

    for model in MODELS.keys():
        print(f"\n{model.upper()}:")
        model_data = df[df['model'] == model]
        
        tones = list(POLITENESS_PREFIXES.keys())
        
        for i, tone1 in enumerate(tones):
            for tone2 in tones[i+1:]:
                t1_scores = model_data[model_data['tone'] == tone1]['correct'].astype(int)
                t2_scores = model_data[model_data['tone'] == tone2]['correct'].astype(int)
                
                if len(t1_scores) > 0 and len(t2_scores) > 0:
                    t_stat, p_value = stats.ttest_ind(t1_scores, t2_scores)
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"  {tone1:15} vs {tone2:15}: p={p_value:.4f} {sig}")
    
    # Sample failed extractions
    print("\nSAMPLE FAILED EXTRACTIONS (first 5):\n")
    failed = df[df['extraction_failed'] == True].head(5)
    for idx, row in failed.iterrows():
        print(f"Model: {row['model']}")
        print(f"Response: '{row['response_raw'][:100]}'")
        print(f"Strategy: {row['extraction_strategy']}")
        print()
    
    df.to_csv("debug_all_results.csv", index=False)

    print("\nAnalysis complete!")
    print("\nFiles saved:")
    print("  - debug_accuracy_overall.csv")
    print("  - debug_extraction_strategies.csv")
    print("  - debug_extraction_failures.csv")
    print("  - debug_accuracy_by_category.csv")
    print("  - debug_all_results.csv")
    print("  - debug_run_N.csv (for each run)")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    # Set up logging to both console and file
    tee = TeeOutput("experiment_log.txt")
    sys.stdout = tee

    try:
        asyncio.run(run_experiment())
    finally:
        # Restore original stdout and close log file
        sys.stdout = tee.terminal
        tee.close()
        print("Log saved to experiment_log.txt")