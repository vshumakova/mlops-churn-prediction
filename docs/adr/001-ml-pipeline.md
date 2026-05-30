# ADR-001: ML Pipeline Architecture

## Context
Need to choose ML system architecture for churn prediction

## Decision
Use Level 2 MLOps with:
- Git + GitHub Actions
- FastAPI + Docker
- Render cloud deployment

## MDD Analysis
- H0: New model is not better than current (ROC-AUC <= 0.75)
- H1: New model is better (ROC-AUC > 0.75)
- Test: paired t-test, p-value = 0.05
- Result: Reject H0 (p-value = 0.001)
- Decision: Deploy new model with automatic rollback

## Status
Accepted
