# Synthetic Incident-Priority Dataset

The capstone uses a synthetic dataset so the complete repository can be shared publicly without exposing employer or client information.

| Column | Type | Meaning | Available at decision time? |
|---|---|---|---|
| incident_id | string | Synthetic incident identifier | Yes |
| timestamp | datetime | Incident creation time | Yes |
| service_criticality | integer 1-5 | Business criticality of the affected service | Yes |
| customer_impact_pct | float | Estimated percentage of customers affected | Yes, subject to measurement error |
| affected_services | integer | Number of services showing impact | Yes |
| error_rate_pct | float | Observed error rate near incident creation | Yes |
| latency_increase_pct | float | Relative latency increase | Yes |
| recent_deploy | binary | Whether a recent deployment may be related | Yes |
| repeat_incident_30d | binary | Similar incident occurred in the previous 30 days | Yes |
| monitoring_confidence | float 0-1 | Confidence in telemetry quality | Yes |
| is_p1 | binary target | Whether the incident became a P1 incident | Known after triage; target only |

## Important limitations

- This is a teaching dataset, not a validated operational dataset.
- The labels are generated from a known probability function and contain randomness.
- Production work would require label definitions, data contracts, privacy review, segment analysis, and monitoring.
- Do not treat performance on this dataset as evidence of production readiness.
