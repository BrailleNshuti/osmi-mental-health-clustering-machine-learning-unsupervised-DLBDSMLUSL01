# Employee Mental-Health Segmentation using Unsupervised Learning
## Project Overview
This project aims to develop a full unsupervised machine-learning pipeline to segment employees of the technology sector according to their mental health attitudes, workplace experiences, disclosure behaviour, and perceived employer support.

The project utilises **the 2016 OSMI Mental Health in Tech Survey** and implements clustering techniques to discover distinct employee profiles, thereby enabling ethical, anonymised, and action-oriented human resource (HR) decisions.

For the analysis, only employed respondents are selected as the survey has different question blocks for employed and self-employed participants. This way, the study will capture genuine workplace mental-health patterns instead of artificial clusters that arise because of the survey structure.


---

# Main Features

* 2016 OSMI Mental Health in Tech Survey dataset
* 1,433 original survey responses
* Focus on 1,146 employed respondents
* Exploratory Data Analysis (EDA)
* Survey-branching structure analysis
* Data-quality cleaning
* Impossible age-value correction
* Free-text gender normalization
* High-cardinality and sparse-column handling
* Treatment-seeking variable held out for validation
* Domain feature engineering using a mental-health support score
* Numeric and categorical preprocessing
* One-hot encoding and scaling
* Label-free feature selection
* UMAP dimensionality reduction
* PCA cross-check
* K-Means clustering
* Gaussian Mixture Model clustering
* Agglomerative clustering
* Internal clustering validation metrics
* Seed-stability analysis
* HR-focused cluster interpretation
* Ethical safeguards and implementation plan

---

# Technologies and Libraries Used

| Technology / Library | Purpose                                                                   |
| -------------------- | ------------------------------------------------------------------------- |
| Python               | Main programming language                                                 |
| Jupyter Notebook     | Full implementation and reproducible workflow                             |
| pandas               | Data loading, cleaning, and manipulation                                  |
| NumPy                | Numerical processing                                                      |
| scikit-learn         | Preprocessing, feature selection, PCA, clustering, and evaluation metrics |
| umap-learn           | Non-linear dimensionality reduction                                       |
| Matplotlib           | Data visualization                                                        |
| Seaborn              | Statistical plots and visualizations                                      |

---

# Dataset

The project uses the publicly available:

**OSMI Mental Health in Tech Survey 2016**

The dataset includes responses about:

* Demographics
* Employment status
* Mental-health history
* Workplace support
* Disclosure comfort
* Employer benefits
* Treatment-seeking behaviour
* Perceived consequences of discussing mental health at work

Dataset source:

https://www.kaggle.com/osmi/mental-health-in-tech-2016

---

# Project Workflow

The unsupervised-learning pipeline follows these stages:

1. Import libraries
2. Load the OSMI 2016 survey dataset
3. Inspect dataset structure and missing values
4. Identify survey branching between employed and self-employed respondents
5. Filter the dataset to employed respondents
6. Hold out the treatment-seeking variable for later validation
7. Clean impossible age values
8. Normalize the free-text gender column
9. Drop self-employed-only, high-cardinality, and sparse columns
10. Engineer a mental-health support score
11. Impute missing values
12. Scale numeric features
13. One-hot encode categorical features
14. Apply label-free feature selection
15. Reduce dimensionality using UMAP
16. Build a PCA representation for cross-checking
17. Compare K-Means, GMM, and Agglomerative clustering
18. Evaluate k values from 2 to 6
19. Compare internal validation metrics
20. Interpret the k = 2 and k = 4 models
21. Validate clusters using the held-out treatment-seeking variable
22. Perform seed-stability checks
23. Translate cluster profiles into HR recommendations
24. Discuss limitations, ethical safeguards, and future work

---

# Models Used

| Model                    | Type                                        |
| ------------------------ | ------------------------------------------- |
| K-Means                  | Partition-based clustering                  |
| Gaussian Mixture Model   | Probabilistic clustering                    |
| Agglomerative Clustering | Hierarchical clustering                     |
| PCA                      | Linear dimensionality-reduction cross-check |
| UMAP                     | Non-linear dimensionality reduction         |

---

# Evaluation Metrics

The clustering solutions were evaluated using:

| Metric                          | Purpose                                                               |
| ------------------------------- | --------------------------------------------------------------------- |
| Silhouette Score                | Measures cluster cohesion and separation                              |
| Davies-Bouldin Index            | Measures average cluster similarity; lower is better                  |
| Calinski-Harabasz Index         | Measures between-cluster separation relative to within-cluster spread |
| PCA-space Silhouette            | Cross-checks whether UMAP separation is overly optimistic             |
| Seed Stability                  | Checks whether results depend strongly on random initialization       |
| Held-out Treatment-Seeking Rate | Interprets whether clusters align with an unseen survey variable      |

---

# Final Results

## Internal Validation Results

| Model       | Silhouette | Davies-Bouldin | Calinski-Harabasz | Cluster Sizes        |
| ----------- | ---------- | -------------- | ----------------- | -------------------- |
| K-Means k=2 | 0.8624     | 0.136          | 8866.8            | 1015 / 131           |
| K-Means k=4 | 0.6845     | 0.364          | 17391.8           | 493 / 131 / 85 / 437 |

## PCA Cross-Check

| Model       | UMAP-space Silhouette | PCA-space Silhouette |
| ----------- | --------------------- | -------------------- |
| K-Means k=2 | 0.8624                | 0.3270               |
| K-Means k=4 | 0.6845                | 0.3736               |

---

# Final Cluster Interpretation

The project reports two complementary clustering solutions:

## Primary Model: K-Means, k = 2

The two-cluster model produces the strongest internal-validation score. It gives a clean statistical split between respondents who generally report a diagnosed or active mental-health condition and those who do not.

## Action-Oriented Model: K-Means, k = 4

The four-cluster model is more useful for HR interpretation because it reveals four actionable employee profiles.

| Cluster | Profile               | Treatment-Seeking Rate | HR Interpretation                                                                             |
| ------- | --------------------- | ---------------------- | --------------------------------------------------------------------------------------------- |
| C0      | Diagnosed & Supported | 92.1%                  | Employees with a condition who are comfortable disclosing and using available support         |
| C1      | Healthy Newcomers     | 45.0%                  | Younger or newer employees with lower reported mental-health burden                           |
| C2      | Affected but Guarded  | 75.3%                  | Employees with a condition who are less comfortable disclosing and fear negative consequences |
| C3      | Low-Need Established  | 18.3%                  | Employees with lower reported mental-health need and more workplace experience                |

The most important HR insight is the contrast between **C0: Diagnosed & Supported** and **C2: Affected but Guarded**. Both groups report mental-health conditions, but they differ strongly in disclosure comfort. This suggests that psychological safety, rather than simply benefit availability, is a key intervention point.

---

# Screenshots of Notebook Outcomes

**Pipeline Overview**
![pipelineoverview](screenshots/pipelineoverview.png)

**Age Distribution Before and After Cleaning**
![agedistribution](screenshots/agedistribution.png)

**UMAP 2-D Embedding Before Clustering**
![umapembedding](screenshots/umapembedding.png)

**Silhouette vs Calinski-Harabasz Across k**
![metriccomparison](screenshots/metriccomparison.png)

**K-Means k=2 Clustering Result**
![kmeansk2](screenshots/kmeansk2.png)

**K-Means k=4 Clustering Result**
![kmeansk4](screenshots/kmeansk4.png)

**Treatment-Seeking Rate by k=4 Cluster**
![treatmentrate](screenshots/treatmentrate.png)

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/BrailleNshuti/osmi-mental-health-clustering-machine-learning-unsupervised-DLBDSMLUSL01
```

## 2. Open the notebook

Open the project in:

* Jupyter Notebook
* JupyterLab
* VS Code
* Google Colab

## 3. Install the required packages

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the notebook cells sequentially from top to bottom.

The notebook includes:

* Data loading
* Data exploration
* Survey-branching analysis
* Data cleaning
* Feature engineering
* Preprocessing
* Feature selection
* UMAP dimensionality reduction
* PCA cross-check
* Clustering algorithm comparison
* Cluster validation
* Cluster interpretation
* HR recommendation development
* Limitations and ethical reflection

---

# Reproducibility

I used a fixed random seed throughout the notebook so that the results can be repeated as closely as possible.

The seed is used for the main steps that involve randomness, including UMAP and the clustering models. Since UMAP can still change slightly depending on the machine or environment, I also added a seed-stability check to see whether the results remain similar across different runs.

---

# Limitations

* The clusters are created in a 2-D UMAP space, so they should not be treated as perfect representations of the original high-dimensional data.
* The PCA check shows weaker separation than UMAP, which means the clusters are useful for exploration but should not be interpreted as fixed natural categories.
* The treatment-seeking question was not used during clustering, but it still comes from the same survey, so it is a helpful check rather than a fully external validation.
* The survey data is self-reported and from 2016, so the exact percentages may not reflect the current technology workplace.
* The analysis only uses employed respondents because self-employed respondents answered a different set of questions.
* The gender column was simplified for modelling, which makes the data easier to use but removes some detail from the original responses.


---

# Ethical Safeguards

Because the project deals with sensitive mental-health data, the results should only be used carefully and responsibly.

Important safeguards:

* Use only anonymised and aggregated survey data.
* Do not attach cluster labels to named employees.
* Do not use clusters for hiring, promotion, performance management, or individual decisions.
* Use the segmentation only to guide supportive resources.
* Communicate clearly to employees how anonymous survey data is used.
* Treat cluster membership as indicative, not definitive.

---

# Future Improvements

The project could be extended by:

* Applying the pipeline to more recent OSMI survey waves
* Building a separate model for self-employed respondents
* Adding a formal data-protection review
* Testing other dimensionality-reduction methods
* Comparing clustering directly in higher-dimensional space
* Running additional cluster-stability experiments
* Building a dashboard for anonymous HR-level monitoring
* Exploring explainability methods for cluster interpretation

---

# Repository Structure

```
employee-mental-health-segmentation/
│
├── screenshots/
│   ├── pipeline_overview.png
│   ├── age_distribution_cleaning.png
│   ├── umap_embedding.png
│   ├── metric_comparison.png
│   ├── kmeans_k2.png
│   ├── kmeans_k4.png
│   └── treatment_rate_k4.png
│
├── data/
│   └── mental_health_survey.csv
│
│   
│
├── Mental_Health_Unsupervised_Braille_4243333.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Reflection

This project helped me understand that clustering is not only about getting separated groups on a plot. The difficult part is deciding whether those groups are meaningful, stable, and useful for a real problem.

The main lesson from this work is that unsupervised learning needs careful interpretation. A good-looking UMAP plot is not enough by itself. The clusters need to be checked with validation metrics, stability tests, comparison against PCA, and a variable that was not used during clustering.

The project also showed that technical results must be handled carefully when the data is sensitive. In this case, the clusters should only be used in an anonymous and aggregated way to guide support, not to label individual employees.


---

# Author

**RUSINGIZA NSHUTI BRAILLE**

Applied Artificial Intelligence, B.Sc.

IU International University of Applied Sciences

---

# Repository Link

https://github.com/BrailleNshuti/osmi-mental-health-clustering-machine-learning-unsupervised-DLBDSMLUSL01