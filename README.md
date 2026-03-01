# Sound to Speech Processing Pipeline

This project implements an automated speech-processing pipeline for acoustic analysis of Tangkhul vowels.  
It extracts vowel formants (F1 and F2) from Praat TextGrid annotations, applies speaker normalization, and visualizes vowel space distributions.

The system is designed to be modular, reproducible, and scalable for cross-speaker or cross-dialect acoustic analysis.

---

## Features

- Automated extraction of vowel segments from Praat TextGrids
- Formant estimation using Burg algorithm (via Parselmouth)
- Speaker-wise Lobanov normalization
- Statistical aggregation of vowel data
- Vowel space visualization with standard deviation ellipses
- Export of raw and normalized datasets

---

## Tech Stack

- Python
- Praat
- Elan
- Parselmouth (Praat interface)
- NumPy
- Pandas
- Matplotlib

---

## Pipeline Workflow

1. Load WAV files and corresponding TextGrid annotations
2. Extract vowel intervals from specified tier
3. Compute F1 and F2 at vowel midpoint
4. Filter unrealistic formant values
5. Apply Lobanov normalization per speaker
6. Aggregate statistics
7. Generate raw and normalized vowel space plots

---


