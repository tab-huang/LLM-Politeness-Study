# LLM Politeness Study: Does Prompt Tone Affect Accuracy?

An empirical study investigating whether the politeness level of prompts affects the accuracy of Large Language Models on standardized knowledge benchmarks.

## Overview

This research tests whether adding polite (or rude) phrasing to prompts impacts model performance on the MMLU-Pro benchmark. We compare 5 politeness levels across 3 leading models over multiple runs.

**Key Finding:** Prompt politeness has minimal to no statistically significant effect on model accuracy.

## Methodology

### Experimental Design
- **Dataset:** MMLU-Pro (100 questions evenly distributed across 14 categories)
- **Models Tested:**
  - GPT-5.2 (OpenAI)
  - Claude Sonnet 4.5 (Anthropic)
  - Gemini-3 Flash Preview (Google)
- **Politeness Levels:** 5 (Very Polite, Polite, Neutral, Rude, Very Rude)
- **Runs:** 5 independent runs per configuration
- **Total API Calls:** 7,500 (100 questions × 5 tones × 3 models × 5 runs)

### Politeness Variations

Each question was prefixed with one of the following tones:

- **Very Polite:** "I would be most grateful if you could kindly assist with this question..."
- **Polite:** "Please answer the following question..."
- **Neutral:** (no prefix)
- **Rude:** "Try not to mess this up..."
- **Very Rude:** "Even you should be able to handle this simple question..."

## Results

### Overall Accuracy by Model and Tone

| Model | Very Polite | Polite | Neutral | Rude | Very Rude |
|-------|-------------|--------|---------|------|-----------|
| **GPT-5.2** | 84.2% | 85.6% | 85.6% | 85.8% | 84.0% |
| **Sonnet 4.5** | 79.4% | 79.0% | 77.4% | 73.4% | 74.0% |
| **Gemini-3** | 75.4% | 76.2% | 76.4% | 75.0% | 75.8% |

### Key Findings

1. **GPT-5.2:** Highest overall accuracy (~85%), virtually no sensitivity to prompt tone (±1.8% variance)
2. **Claude Sonnet 4.5:** Moderate accuracy (~77%), slight preference for polite tones (~6% difference between best and worst)
3. **Gemini-3:** Moderate accuracy (~76%), minimal tone sensitivity (±1.4% variance)
4. **Statistical Significance:** Most tone comparisons showed no significant differences (p > 0.05)

## Files

### Code
- `LLM_Politeness_Test_Script_3.py` - Main experiment script with enhanced extraction logging

### Results
- `debug_accuracy_overall.csv` - Accuracy summary by model and tone
- `debug_accuracy_by_category.csv` - Accuracy broken down by MMLU category
- `debug_all_results.csv` - Complete results dataset
- `debug_extraction_strategies.csv` - Analysis of answer extraction methods
- `debug_extraction_failures.csv` - Extraction failure rates
- `debug_run_1.csv` through `debug_run_5.csv` - Results from individual runs
- `experiment_log.txt` - Complete execution log with debug output

### Archive
- `OLD/` - Previous versions and intermediate results

## Setup and Usage

### Prerequisites

```bash
pip install openai pandas datasets scipy
```

### Configuration

Set your OpenRouter API key as an environment variable:

```bash
# Linux/Mac
export OPENROUTER_API_KEY="your-key-here"

# Windows (PowerShell)
$env:OPENROUTER_API_KEY="your-key-here"

# Windows (Command Prompt)
set OPENROUTER_API_KEY=your-key-here
```

### Running the Experiment

```bash
python LLM_Politeness_Test_Script_3.py
```

**Note:** The full experiment makes 7,500 API calls and costs approximately $10-15, taking 2-3 hours to complete.

### Configuration Options

Edit the following constants in the script to customize:

```python
NUM_RUNS = 5           # Number of independent runs
NUM_QUESTIONS = 100    # Questions per run (max 100)
DEBUG_MODE = True      # Enable detailed logging
```

## Technical Details

### Answer Extraction

The script uses a sophisticated multi-strategy extraction system:

1. **Strategy 0:** Reasoning conclusion detection (for models that explain their answers)
2. **Strategy 1:** Exact single letter match
3. **Strategy 2:** Letter with punctuation (e.g., "A." or "A)")
4. **Strategy 3:** "Answer is X" pattern matching
5. **Strategy 4-5:** First/last line or word extraction
6. **Strategy 6:** Context-aware search (avoids negative contexts)
7. **Strategy 7:** Desperate fallback (any valid letter found)

### Statistical Analysis

The script performs:
- Overall accuracy calculations
- Per-category breakdowns
- Extraction failure rate analysis
- Two-sample t-tests for statistical significance between tone levels

## Interpretation

This study suggests that **prompt engineering focused on politeness is not a significant factor** for improving accuracy on factual knowledge tasks. Models respond consistently to the semantic content of questions regardless of tone.

### Implications:
- **For developers:** Focus on clear, well-structured prompts rather than tone
- **For researchers:** Other prompt engineering factors (formatting, examples, chain-of-thought) likely have greater impact
- **For users:** Natural language interaction works well; no need to be overly polite or worry about tone

## Limitations

- Single benchmark (MMLU-Pro) - results may differ for creative or subjective tasks
- Specific prompt phrasings - other politeness formulations might have different effects
- Models tested at specific versions - behavior may change with updates
- 100 questions - larger sample sizes could reveal subtle effects

## Future Work

- Test on creative tasks (storytelling, poetry, brainstorming)
- Examine multilingual prompts and cultural variations
- Investigate combined effects with other prompt engineering techniques
- Test on smaller/distilled models vs. frontier models

## Citation

If you use this research, please cite:

```
LLM Politeness Study (2025)
Research on prompt tone effects on MMLU-Pro benchmark accuracy
https://github.com/[your-username]/LLM_Politeness
```

## License

This research is provided as-is for educational and research purposes. The MMLU-Pro dataset has its own license from TIGER-Lab.

## Acknowledgments

- MMLU-Pro dataset by TIGER-Lab
- OpenRouter for unified API access
- OpenAI, Anthropic, and Google for model access
