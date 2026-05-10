# LLM-AHP-MCDA: Evaluating Large Language Models as Preference Elicitation 
# Instruments in Multi-Criteria Decision Analysis

**MSc Dissertation Research**  
Rafi Khandoker | University of Leeds  
MSc Business Analytics and Decision Science | 2025–2026

---

## Overview

This repository contains the code and analysis framework developed for my MSc 
dissertation, which investigates whether large language models (LLMs) — 
specifically GPT-4 and Claude — can generate pairwise comparison matrices that 
are internally consistent and aligned with human expert judgements within the 
Analytic Hierarchy Process (AHP) framework.

AHP is a widely-used multi-criteria decision analysis (MCDA) method that 
structures complex decisions by breaking them into pairwise comparisons. 
Traditionally, these comparisons are elicited from human domain experts. This 
research asks: can LLMs serve as a substitute or supplement to human experts?

---

## Research Questions

1. Can GPT-4 and Claude generate pairwise comparison matrices that satisfy 
   AHP internal consistency thresholds (CR < 0.1)?
2. How closely do LLM-generated matrices align with expert-elicited matrices 
   from published literature?
3. Does prompt design affect consistency and alignment outcomes?

---

## Repository Structure

---

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/rafi-khandoker/llm-ahp-mcda.git
cd llm-ahp-mcda
pip install -r requirements.txt
```

---

## Key Concepts

**Pairwise Comparison Matrix:** An n×n matrix where each entry a_ij represents 
how much criterion i is preferred over criterion j, using Saaty's 1–9 scale.

**Consistency Ratio (CR):** A measure of logical consistency in judgements.  
CR = CI / RI, where CI is the Consistency Index and RI is the Random Index.  
A CR < 0.1 indicates acceptable consistency.

**Priority Vector:** The normalised principal eigenvector of the comparison 
matrix, representing the relative weights of criteria.

---

## Usage

```python
from src.ahp import AHPMatrix

# Define a pairwise comparison matrix
matrix = [
    [1,   3,   5],
    [1/3, 1,   2],
    [1/5, 1/2, 1]
]

ahp = AHPMatrix(matrix)
print(f"Priority Vector: {ahp.priority_vector()}")
print(f"Consistency Ratio: {ahp.consistency_ratio():.4f}")
print(f"Is consistent (CR < 0.1): {ahp.is_consistent()}")
```

---

## Citation

If you use this code in your research, please cite:
Khandoker, R. (2026). Investigating Large Language Models as Preference
Elicitation Instruments in Multi-Criteria Decision Analysis.
