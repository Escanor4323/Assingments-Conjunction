# C2: Surname Likelihood and Generation

This folder contains the implementation and results for the C2 assignment.

## Files

-   `analysis.py`: Python script that:
    -   Trains a character-level bigram model on English names.
    -   Estimates the likelihood of specific surnames.
    -   Generates name completions for specific prefixes.
-   `report.md`: A report summarizing the findings, including the likelihood table and model critique.

## Usage

To run the analysis script, navigate to this directory and run:

```bash
python3 analysis.py
```

Note: The script expects the `data` directory to be in the project root (one level up), and `utils.py` to be in `../src`.
