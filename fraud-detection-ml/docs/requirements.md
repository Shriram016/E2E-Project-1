
# **Project 1 – Requirements**

## **1. Functional Requirements**

1. **Data Ingestion**

   * Read raw transaction data from CSV/Parquet (`data/raw/`).
   * Validate data integrity: unique transaction IDs, valid timestamps, non-negative amounts.
   * Handle missing and inconsistent values.

2. **Data Preprocessing**

   * Clean and normalize features: numeric scaling, categorical encoding.
   * Feature engineering:

     * Temporal features (hour, day, transaction frequency)
     * Aggregated statistics per user/device
     * Ratio features (amount vs user average, etc.)
   * Save processed data in `data/processed/` for modeling.

3. **Modeling**

   * Train ML models capable of handling **highly imbalanced datasets**.
   * Support **ensemble models** (LightGBM, XGBoost, Random Forest).
   * Evaluate models with **precision, recall, F1-score, PR-AUC**, not just accuracy.

4. **Deployment / API**

   * REST API for real-time predictions using FastAPI.
   * Input: transaction data (JSON).
   * Output: predicted class, probability, model version, optional explainability.
   * Dockerized deployment for cloud readiness.

5. **Monitoring & Explainability**

   * Track model performance, feature distributions, and data drift.
   * Integrate **SHAP/LIME** for interpretability.
   * Alerting mechanism for drift or performance degradation.

---

## **2. Non-Functional Requirements**

1. **Reproducibility**

   * Versioned datasets, environment, and scripts.
   * Use virtualenv or conda for environment isolation.

2. **Scalability & Performance**

   * Able to handle \~1.5M+ transactions efficiently.
   * Predict transactions in **<500ms per request** (simulated).

3. **Code Quality & Maintainability**

   * Modular, testable, well-documented code.
   * Unit tests and CI/CD setup for automated validation.

4. **Compliance & Security**

   * Sensitive info anonymized or removed.
   * Audit logs for predictions and model decisions.

5. **Robustness & Reliability**

   * Handle missing or corrupted input gracefully.
   * Monitor for model drift and retrain as needed.

---

## **3. Success Criteria**

* Data pipeline is reproducible from raw → processed.
* Model achieves high PR-AUC and handles class imbalance.
* API serves real-time predictions reliably.
* Monitoring dashboards and explainability integrated.
* Documentation complete for future maintainers.

