# 🔬 PRISM-MB Project

## Probabilistic Response-factor Informed Structural Mass Balance Framework

**NEST 2.0 - Problem Statement 3**  
Mass Balance Calculation Methods Evaluation in Analytical Forced Degradation Studies

---
```
## 📁 Project Structure
PRISM-MB-Project/
│
├── webapp/ # Streamlit Web Application (Render Deployment)
│ ├── app.py
│ ├── requirements.txt
│ ├── render.yaml
│ └── README.md
│
├── research/ # Research & Validation Notebooks
│ ├── 01_PRISM_Core_Framework.ipynb
│ ├── 02_PRISM_Validation.ipynb
│ ├── figures/
│ └── data/
│
├── presentation/ # Final Submission Materials
│ ├── PRISM_MB_Presentation.pptx
│ └── assets/
│
└── README.md # This file
```


---

## 🚀 Live Demo

**Web Application:** [https://prism-mb.onrender.com](https://prism-mb.onrender.com)

---

## 💡 The Innovation

### The Problem
Current mass balance methods in pharmaceutical forced degradation studies:
1. **Ignore response factor differences** between API and degradants
2. **Provide point estimates** with no uncertainty quantification
3. **Use arbitrary pass/fail thresholds** without scientific justification

### The Solution: PRISM-MB
1. **Response Factor Correction (RFCMB)** - Mathematically corrects for detector response differences
2. **Monte Carlo Uncertainty Quantification** - 10,000 simulations provide probability distributions
3. **Risk-Based Decision Engine** - Probabilistic thresholds for Accept/Investigate/Revise

---

## 📈 Validation Results

Tested on 200 synthetic scenarios with known ground truth:

| Method | Mean Absolute Error | Improvement |
|--------|--------------------:|------------:|
| AMB (Conventional) | 9.8% | Baseline |
| RFCMB (PRISM) | 4.2% | **+57%** |
| DAMB (PRISM) | 3.5% | **+64%** |

Statistical significance: **p < 0.001**

---

## 🛠️ Technology Stack

- **Framework:** Streamlit
- **Computation:** NumPy, SciPy, Pandas
- **Visualization:** Plotly
- **Deployment:** Render
- **Research:** Jupyter Notebooks

---

## 📋 How to Run Locally

### Web App
```
cd webapp
pip install -r requirements.txt
streamlit run app.py
Notebooks
Bash

cd research
jupyter notebook
```
```
👨‍💻 Author
Aryan Ranjan
B.Tech CSE (E-Commerce Technologies)
VIT Bhopal University

📜 License
MIT License

🏆 NEST 2.0
This project addresses Problem Statement 3: Mass Balance Calculation Methods Evaluation in Analytical Forced Degradation Studies

Tracks Covered:

✅ Track 1: Literature-Based Formula Optimization
✅ Track 2: Computational Validation

"Transforming pharmaceutical mass balance from deterministic guesswork to probabilistic intelligence."
