"""
Prompt Politeness LLM Accuracy Study - MMLU-Pro Version
100 questions × 5 politeness levels × 3 models × 5 runs = 7,500 API calls
Estimated cost: $10-15, Runtime: 2-3 hours
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

NUM_RUNS = 2  # Reduced from 10
NUM_QUESTIONS = 10  # Increased from 50
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
    Load 100 questions from MMLU-Pro evenly distributed across subjects
    MMLU-Pro has 14 categories, so ~7 questions per category
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
# ANSWER EXTRACTION (handles A-J for MMLU-Pro)
# ============================================================================

def extract_letter(response, num_options):
    """
    Multi-strategy answer extraction
    Handles A-J based on number of options in question
    """
    if not response:
        return None
        
    response = response.strip().upper()
    
    # Valid letters based on number of options
    valid_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][:num_options]
    
    # Strategy 1: Just a single letter
    if response in valid_letters:
        return response
    
    # Strategy 2: Common patterns
    patterns = [
        r'^([A-J])\.?\s*$',
        r'ANSWER:\s*([A-J])',
        r'THE ANSWER IS\s*([A-J])',
        r'CORRECT ANSWER IS\s*([A-J])',
        r'\*\*([A-J])\*\*',
        r'\b([A-J])\s*IS CORRECT',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match and match.group(1) in valid_letters:
            return match.group(1)
    
    # Strategy 3: First valid letter (avoid negative contexts)
    negative_keywords = ['NOT', 'INCORRECT', 'WRONG', 'ELIMINATE', 'EXCEPT']
    letters = re.findall(r'\b([A-J])\b', response)
    
    # Filter to only valid letters
    valid_found = [l for l in letters if l in valid_letters]
    
    if valid_found:
        first_letter = valid_found[0]
        letter_pos = response.find(first_letter)
        context = response[max(0, letter_pos-30):letter_pos+30]
        
        if any(kw in context for kw in negative_keywords) and len(valid_found) > 1:
            return valid_found[1]
        
        return first_letter
    
    return None

# ============================================================================
# API CALL
# ============================================================================

async def call_model(question_text, model_name, num_options):
    """Call model with improved memory clearing"""
    try:
        response = await client.chat.completions.create(
            model=MODELS[model_name],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You MUST respond with ONLY a single letter. "
                        f"Valid answers: A, B, C, D, E, F, G, H, I, J (up to {chr(64+num_options)}). "
                        "DO NOT explain your reasoning. "
                        "DO NOT write 'I think' or 'The answer is'. "
                        "DO NOT provide any text except the letter. "
                        "WRONG: 'I need to calculate...' "
                        "WRONG: 'The answer is B because...' "
                        "CORRECT: 'B' "
                        "Output ONLY the letter, nothing else."
                    )
                },
                {
                    "role": "user", 
                    "content": (
                        "CRITICAL: Answer with ONLY ONE LETTER. No explanations.\n\n"
                        + question_text + "\n\n"
                        f"Your response must be exactly 1 character: A-{chr(64+num_options)}"
                    )
                }
            ],
            temperature=0.0,  # Changed from 0.1 to be more deterministic
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"❌ API Error for {model_name}: {e}")
        return None

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

async def run_experiment():
    """Run the full experiment"""
    
    print("="*70)
    print("PROMPT POLITENESS ON MMLU-PRO")
    print("100 questions × 5 tones × 3 models × 5 runs = 7,500 calls")
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
    print(f"Estimated cost: $10-15")
    print(f"Estimated time: 2-3 hours\n")
    
    start_time = datetime.now()
    
    # Run experiment
    for run in range(NUM_RUNS):
        print(f"\n{'='*70}")
        print(f"RUN {run + 1}/{NUM_RUNS}")
        print(f"{'='*70}\n")
        
        run_start = datetime.now()
        
        for model_name in MODELS.keys():
            print(f"Testing {model_name}...")
            
            for tone_name, prefixes in POLITENESS_PREFIXES.items():
                
                tasks = []
                question_meta = []
                
                for q in formatted_questions:
                    prefix = random.choice(prefixes)
                    full_prompt = prefix + q['prompt']
                    
                    tasks.append(call_model(full_prompt, model_name, q['num_options']))
                    question_meta.append({
                        'qid': q['qid'],
                        'category': q['category'],
                        'correct': q['correct'],
                        'tone': tone_name,
                        'model': model_name,
                        'run': run,
                        'num_options': q['num_options']
                    })
                
                responses = await asyncio.gather(*tasks)
                
                for meta, response in zip(question_meta, responses):
                    extracted = extract_letter(response, meta['num_options']) if response else None
                    is_correct = (extracted == meta['correct']) if extracted else False
                    
                    result = {
                        **meta,
                        'response_raw': response,
                        'extracted': extracted,
                        'correct': is_correct,
                        'extraction_failed': extracted is None,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    all_results.append(result)
                    current_call += 1
                
                print(f"  {tone_name}: {current_call}/{total_calls} calls")
                await asyncio.sleep(SLEEP_BETWEEN_BATCHES)
        
        run_elapsed = (datetime.now() - run_start).total_seconds()
        total_elapsed = (datetime.now() - start_time).total_seconds()
        remaining_runs = NUM_RUNS - (run + 1)
        estimated_remaining = (total_elapsed / (run + 1)) * remaining_runs
        
        print(f"\nRun {run+1} completed in {run_elapsed/60:.1f} minutes")
        print(f"Total elapsed: {total_elapsed/60:.1f} minutes")
        print(f"Estimated remaining: {estimated_remaining/60:.1f} minutes")
        
        # Save intermediate results after each run
        pd.DataFrame(all_results).to_csv(f"mmlu_pro_run_{run+1}.csv", index=False)
    
    print("\n" + "="*70)
    print("ANALYZING RESULTS")
    print("="*70)
    
    analyze_results(all_results)

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_results(all_results):
    """Analyze and save results with category breakdowns"""
    
    df = pd.DataFrame(all_results)
    
    # Overall accuracy by model and tone
    print("\n📊 OVERALL ACCURACY\n")
    summary = df.groupby(['model', 'tone'])['correct'].agg([
        ('accuracy_%', lambda x: (x.sum() / len(x)) * 100),
        ('correct_count', 'sum'),
        ('total', 'count')
    ]).round(2)
    
    print(summary)
    summary.to_csv("mmlu_pro_accuracy_overall.csv")
    
    # Accuracy by category
    print("\n📚 ACCURACY BY CATEGORY\n")
    category_summary = df.groupby(['model', 'tone', 'category'])['correct'].agg([
        ('accuracy_%', lambda x: (x.sum() / len(x)) * 100),
    ]).round(2)
    
    print(category_summary)
    category_summary.to_csv("mmlu_pro_accuracy_by_category.csv")
    
    # Statistical tests
    print("\n📈 STATISTICAL SIGNIFICANCE (t-tests, α=0.05)\n")
    
    from scipy import stats
    
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
    
    # Extraction failure rate
    print("\n⚠️  EXTRACTION FAILURE RATES\n")
    extraction_failures = df.groupby(['model', 'tone'])['extraction_failed'].agg([
        ('failure_rate_%', lambda x: (x.sum() / len(x)) * 100)
    ]).round(2)
    print(extraction_failures)
    
    df.to_csv("mmlu_pro_all_results.csv", index=False)
    
    print("\n✅ Analysis complete!")
    print("\nFiles saved:")
    print("  - mmlu_pro_accuracy_overall.csv")
    print("  - mmlu_pro_accuracy_by_category.csv")
    print("  - mmlu_pro_all_results.csv")
    print("  - mmlu_pro_run_N.csv (for each run)")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    asyncio.run(run_experiment())