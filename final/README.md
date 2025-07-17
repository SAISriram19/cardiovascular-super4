# Final Submission: Automated ECG Arrhythmia Classification

This folder contains the final deliverables for the Automated ECG Arrhythmia Classification project, including the application interface and all model training and preprocessing scripts. The structure and instructions below ensure clarity, reproducibility, and ease of use for both technical and non-technical stakeholders.

## Directory Structure

- `application/`  
  Contains the Streamlit app (`app.py`) for ECG arrhythmia classification. 
  
- `pre_processing and training/`
  - `M-1/` — MIT-BIH Arrhythmia
  - `M-2/` — PTB-XL
  - `M-3/` — Atrial Fibrillation

## Model Generation Workflow

### M-1: MIT-BIH Arrhythmia
- **Step 1:** Run the single file in `M-1/`.
- **Output:** This will generate `model.pkl` and `scaler.pkl` files.

### M-2: PTB-XL
- **Step 1:** Run the preprocessing file in `M-2/`.
- **Step 2:** Run the training file in `M-2/`.
- **Output:** The required `.pkl` files will be generated.

### M-3: Atrial Fibrillation
- **Step 1:** Run `CMPE_255_ECG_Analysis_Sample_FInal (1).ipynb` in `M-3/`.
- **Step 2:** Run `Merging_Scaled_1D_&_Trying_Different_CLassification_ML_Models_.ipynb` in `M-3/`.
- **Output:** The required `.pkl` file will be generated.

## Datasets

| Model | Dataset Name           | Dataset Link (to be added) |
|-------|------------------------|----------------------------|
| M-1   | MIT-BIH Arrhythmia     | [link]                     |
| M-2   | PTB-XL                 | [link]                     |
| M-3   | Atrial Fibrillation    | [link]                     |

*Please update the above table with dataset links as needed.*

## Integration with Application

- After generating all required `.pkl` files, update their file paths in `application/app.py` to ensure the Streamlit app functions correctly.

## Notes & Best Practices

- All code is modular and organized for clarity and maintainability.
- The workflow is designed for reproducibility and ease of handover.
- Please ensure all dependencies are installed as per the project requirements (see main project README for details).

---

For any further clarifications or updates, please refer to the main project README or contact the project maintainers.
