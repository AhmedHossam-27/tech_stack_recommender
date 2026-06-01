# Project 3: AI Recommendation Logic 🚀
### Industrial Training Kit | Batch: 2026 — Powered by DecodeLabs

An AI-driven career and tech stack recommendation engine built as part of the DecodeLabs Artificial Intelligence Engineering Internship. This project transitions from passive classification to active prediction, utilizing pure similarity logic to match user profiles with technical career paths, resolving the "Choice Overload" dilemma.

---

## 📌 Project Overview & Goal
The main objective is to design a **Content-Based Filtering** recommendation engine. The system captures explicit user skills, processes them using advanced mathematical mapping, and returns a tailored **Top-3 List** of matching job roles. 

To ensure professional-grade accuracy, the system avoids simple binary overlaps and instead employs **TF-IDF Vectorization** and **Cosine Similarity** to properly penalize generic tags and reward highly specific technical attributes.

---

## ⚙️ Core Architecture (IPO Model)
The system operates on a clean **Input-Process-Output** data pipeline:
1. **Input (Ingestion):** Captures at least 3 distinct user-defined technical skills or tags.
2. **Process (Scoring & Math Engine):** * **Vector Mapping:** Converts textual raw skills into numerical matrices.
   * **TF-IDF Weighting:** Mathematically adjusts the balance between common phrases and niche domain terms.
   * **Cosine Similarity:** Measures the angular distance between the user profile vector and pre-defined career track vectors.
3. **Output (Filtering):** Fuses the similarity data, sorts the scores in descending order, and displays a refined list of top recommendations to prevent cognitive overload.

---

## 🛠️ Cold Start Strategy
To handle scenarios where a new user provides empty input (resulting in failing zero-vectors), the system integrates a robust **Trending Fallback** mechanism. Instead of raising a runtime error, it catches empty vectors and automatically triggers general trending career recommendations to maintain a seamless onboarding experience.

---

## 🚀 Installation & Setup

### Prerequisites
Make sure you have Python 3.x installed along with `pandas` and `scikit-learn`.

### Installation
Clone the repository or save the file as `tech_stack_recommender.py`, then install the required dependencies using pip:
```bash
pip install pandas scikit-learn
