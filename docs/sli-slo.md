# SLI/SLO Definitions

## 1. Technical SLI

| SLI | SLO |
|-----|-----|
| API latency (p95) | < 500ms |
| API availability | 99.9% |
| Error rate | < 0.1% |
| Health check status | 200 OK |

## 2. Model SLI

| SLI | SLO |
|-----|-----|
| ROC-AUC | > 0.75 |
| Precision | > 0.70 |
| Recall | > 0.70 |
| F1-Score | > 0.70 |
| Model retraining frequency | Weekly (every Sunday) |
| Model version freshness | < 7 days |

## 3. Business SLI

| SLI | SLO |
|-----|-----|
| Churn rate reduction | > 15% per quarter |
| Customer retention increase | > 10% per quarter |

## Monitoring Implementation

- **Retraining:** GitHub Actions cron schedule (0 0 * * 0)
- **Health checks:** `/health` endpoint every 30s
- **Model metrics:** Saved to `metrics/latest_metrics.json`

## Alerting

| Condition | Action |
|-----------|--------|
| ROC-AUC < 0.75 | Trigger retraining |
| API down for 5 min | Alert on-call |
| Drift detected | Review data pipeline |
