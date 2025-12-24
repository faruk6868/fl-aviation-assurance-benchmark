# GSN Assurance Case

This report summarises the Goal Structuring Notation (GSN) assurance case generated from the Concentric Assurance Map (CAM) artefacts.

## Overview

- Top claim: FL-PdM system meets safety and assurance objectives within the intended ODD.
- Pillar coverage: 4 primary assurance goals reflecting data governance, process assurance, robustness & security, and trustworthiness & explainability.
- Requirement goals: 40
- Metric goals: 40
- Evidence nodes: 77

![GSN Assurance Case](GSN.svg)

## Coverage Summary

- Requirements with linked metrics: 40 / 40
- Metrics with evidence: 77 / 40

## Evidence Table

| Evidence ID | Test | Metric | Status | Pass Level | Observed | Unit | Comment |
|-------------|------------|--------|--------|------------|----------|------|---------|
| Sn_T10_0_M13 | T10 | M13 | PASS | Cat | 0.000256459399152 | variance | Meets Cat threshold (margin +0.04974). Per-client accuracy range 0.9375-0.9897. |
| Sn_T10_0_M14 | T10 | M14 | PASS | Cat | 0.0160143497886131 | std | Meets Cat threshold (margin +0.03399). Per-client accuracy range 0.9375-0.9897. |
| Sn_T10_0_M15 | T10 | M15 | PASS | Cat | 0.0014369176318706 | abs_ratio | Meets Cat threshold (margin +0.01856). Group positive-rate spread 0.0104-0.0119. |
| Sn_T10_0_M16 | T10 | M16 | PASS | Haz | 0.0373641304347825 | abs_ratio | Meets Haz threshold (margin +0.01264). Group recall spread 0.4844-0.5217. |
| Sn_T10_0_M17 | T10 | M17 | PASS | Cat | 0.9989158974933292 | ratio | Meets Cat threshold (margin +0.04892). Evaluated across 3 groups (min support 4310). |
| Sn_T10_0_M35 | T10 | M35 | PASS | Cat | 0.000256459399152 | variance | Meets Cat threshold (margin +0.04974). Per-client accuracy range 0.9375-0.9897. |
| Sn_T11_0_M31 | T11 | M31 | FAIL |  | 0.413066416978836 | ratio | Violates Maj threshold by 0.1869 (below required value). Review explainability pipeline and refresh attribution baselines. |
| Sn_T11_0_M32 | T11 | M32 | PASS | Cat | 1.0 | ratio | Meets Cat threshold (margin +0.15). |
| Sn_T12_0_M1 | T12 | M1 | PASS | Haz | 0.9875534514355528 | ratio | Meets Haz threshold (margin +0.02755). |
| Sn_T12_0_M10 | T12 | M10 | PASS | Haz | 60.0 | rounds | Meets Haz threshold (margin +0). |
| Sn_T12_0_M11 | T12 | M11 | PASS | Cat | 0.3910491943359376 | MB | Meets Cat threshold (margin +49.61). |
| Sn_T12_0_M12 | T12 | M12 | FAIL |  | 2.525394415177283 | x | Violates Maj threshold by 2.475 (below required value). Optimise communication budget and scheduling (compression, partial participation). |
| Sn_T12_0_M13 | T12 | M13 | PASS | Cat | 0.0361488955460356 | variance | Meets Cat threshold (margin +0.01385). Per-client accuracy range 0.3913-0.9698. |
| Sn_T12_0_M14 | T12 | M14 | FAIL |  | 0.190128628949024 | std | Violates Maj threshold by 0.09013 (above required value). Per-client accuracy range 0.3913-0.9698. Apply fairness-aware aggregation or reweight under-performing client groups. |
| Sn_T12_0_M15 | T12 | M15 | PASS | Cat | 0.0050256735770649 | abs_ratio | Meets Cat threshold (margin +0.01497). Group positive-rate spread 0.0144-0.0194. |
| Sn_T12_0_M16 | T12 | M16 | PASS | Maj | 0.06462063086104 | abs_ratio | Meets Maj threshold (margin +0.03538). Group recall spread 0.5942-0.6588. |
| Sn_T12_0_M17 | T12 | M17 | PASS | Cat | 0.998028660048374 | ratio | Meets Cat threshold (margin +0.04803). Evaluated across 3 groups (min support 4310). |
| Sn_T12_0_M18 | T12 | M18 | MONITOR | Monitor | 0.0 | relative | Metric monitored without gating threshold. |
| Sn_T12_0_M19 | T12 | M19 | FAIL |  | 3.765529249228086 | nats | Violates Maj threshold by 3.266 (above required value). Mitigate client heterogeneity via reweighting or shared feature normalisation. |
| Sn_T12_0_M2 | T12 | M2 | FAIL |  | 0.6267281105990783 | ratio | Violates Maj threshold by 0.2733 (below required value). Recalibrate decision thresholds or adjust loss weighting to curb false alarms. |
| Sn_T12_0_M20 | T12 | M20 | PASS | Haz | 5.0 | epsilon | Meets Haz threshold (margin +0). |
| Sn_T12_0_M21 | T12 | M21 | FAIL |  | 0.0 | ratio | Violates Maj threshold by 0.85 (below required value). Enable robust aggregation and investigate anomalous client behaviours. |
| Sn_T12_0_M22 | T12 | M22 | PASS | Cat | 0.9983018139714396 | ratio | Meets Cat threshold (margin +0.0783). |
| Sn_T12_0_M23 | T12 | M23 | FAIL |  | 0.0008010387651721 | ratio | Violates Maj threshold by 0.8992 (below required value). Enable robust aggregation and investigate anomalous client behaviours. |
| Sn_T12_0_M24 | T12 | M24 | ALARM | Cat | 0.0 | p_value | Alarm triggered at Cat (value 0 < alpha 0.01). KS statistic 0.2798. Trigger drift response plan and recalibrate with recent operational data. |
| Sn_T12_0_M25 | T12 | M25 | PASS | Cat | 1.0 | ratio | Meets Cat threshold (margin +0.1). |
| Sn_T12_0_M26 | T12 | M26 | PASS | Maj | 0.9 | ratio | Meets Maj threshold (margin +0). |
| Sn_T12_0_M27 | T12 | M27 | PASS | Cat | 0.0016981860285604 | ratio | Meets Cat threshold (margin +0.1483). |
| Sn_T12_0_M28 | T12 | M28 | PASS | Cat | 0.0478459657312108 | ratio | Meets Cat threshold (margin +0.002154). |
| Sn_T12_0_M29 | T12 | M29 | FAIL |  | 1.604189453125 | abs_ratio | Violates Maj threshold by 1.504 (above required value). Refine predictive intervals via ensembling or Bayesian calibration. |
| Sn_T12_0_M3 | T12 | M3 | FAIL |  | 0.6238532110091743 | ratio | Violates Maj threshold by 0.2761 (below required value). Increase recall through threshold tuning or focused retraining on failure cases. |
| Sn_T12_0_M30 | T12 | M30 | PASS | Cat | 149.91001730001153 | seconds | Meets Cat threshold (margin +3450). |
| Sn_T12_0_M31 | T12 | M31 | FAIL |  | 0.231410413980484 | ratio | Violates Maj threshold by 0.3686 (below required value). Review explainability pipeline and refresh attribution baselines. |
| Sn_T12_0_M32 | T12 | M32 | PASS | Cat | 1.0 | ratio | Meets Cat threshold (margin +0.15). |
| Sn_T12_0_M33 | T12 | M33 | PASS | Cat | 1.0 | ratio | Meets Cat threshold (margin +0.01). |
| Sn_T12_0_M34 | T12 | M34 | PASS | Cat | 1.0 | ratio | Meets Cat threshold (margin +0.03). |
| Sn_T12_0_M35 | T12 | M35 | PASS | Cat | 0.0361488955460356 | variance | Meets Cat threshold (margin +0.01385). Per-client accuracy range 0.3913-0.9698. |
| Sn_T12_0_M36 | T12 | M36 | FAIL |  | 0.3 | ratio | Violates Maj threshold by 0.35 (below required value). Optimise communication budget and scheduling (compression, partial participation). |
| Sn_T12_0_M37 | T12 | M37 | PASS | Cat | 0.0444374084472656 | MB | Meets Cat threshold (margin +1024). |
| Sn_T12_0_M38 | T12 | M38 | PASS | Cat | 0.0059164062520267 | ms | Meets Cat threshold (margin +49.99). |
| Sn_T12_0_M39 | T12 | M39 | PASS | Haz | 0.0062897965522596 | ratio | Meets Haz threshold (margin +0.03371). |
| Sn_T12_0_M4 | T12 | M4 | FAIL |  | 0.6252873563218391 | ratio | Violates Maj threshold by 0.2747 (below required value). Balance precision/recall via class weighting or adaptive thresholds. |
| Sn_T12_0_M40 | T12 | M40 | FAIL |  | 0.3761467889908257 | ratio | Violates Maj threshold by 0.2761 (above required value). Increase recall through threshold tuning or focused retraining on failure cases. |
| Sn_T12_0_M5 | T12 | M5 | FAIL |  | 37.69907379150391 | cycles | Violates Maj threshold by 27.99 (above required value). Improve RUL estimator accuracy or extend the proactive maintenance lead time. |
| Sn_T12_0_M6 | T12 | M6 | FAIL |  | 336679.6170726881 | score | Violates Maj threshold by 3.367e+05 (above required value). Improve RUL estimator accuracy or extend the proactive maintenance lead time. |
| Sn_T12_0_M7 | T12 | M7 | PASS | Cat | 0.9897880746768188 | ratio | Meets Cat threshold (margin +0.009788). |
| Sn_T12_0_M8 | T12 | M8 | FAIL |  | 0.6212558883156102 | abs_ratio | Violates Maj threshold by 0.5713 (above required value). Apply regularisation or data augmentation to shrink the generalisation gap. |
| Sn_T12_0_M9 | T12 | M9 | PASS | Maj | 583.2383815595972 | delta_per_round | Meets Maj threshold (margin +1.942e+04). |
| Sn_T1_0_M1 | T1 | M1 | PASS | Haz | 0.9887127372565468 | ratio | Meets Haz threshold (margin +0.02871). |
| Sn_T1_0_M29 | T1 | M29 | FAIL |  | 1.2648448944091797 | abs_ratio | Violates Maj threshold by 1.165 (above required value). Refine predictive intervals via ensembling or Bayesian calibration. |
| Sn_T1_0_M3 | T1 | M3 | FAIL |  | 0.4393351800554017 | ratio | Violates Maj threshold by 0.4607 (below required value). Increase recall through threshold tuning or focused retraining on failure cases. |
| Sn_T1_0_M5 | T1 | M5 | FAIL |  | 16.01161003112793 | cycles | Violates Maj threshold by 6.304 (above required value). Improve RUL estimator accuracy or extend the proactive maintenance lead time. |
| Sn_T1_0_M7 | T1 | M7 | PASS | Cat | 0.9916045270808324 | ratio | Meets Cat threshold (margin +0.0116). |
| Sn_T2_0_M10 | T2 | M10 | PASS | Cat | 50.0 | rounds | Meets Cat threshold (margin +0). |
| Sn_T2_0_M29 | T2 | M29 | FAIL |  | 1.2708920288085936 | abs_ratio | Violates Maj threshold by 1.171 (above required value). Refine predictive intervals via ensembling or Bayesian calibration. |
| Sn_T2_0_M3 | T2 | M3 | FAIL |  | 0.6376146788990825 | ratio | Violates Maj threshold by 0.2624 (below required value). Increase recall through threshold tuning or focused retraining on failure cases. |
| Sn_T2_0_M5 | T2 | M5 | FAIL |  | 16.039268493652344 | cycles | Violates Maj threshold by 6.331 (above required value). Improve RUL estimator accuracy or extend the proactive maintenance lead time. |
| Sn_T2_0_M7 | T2 | M7 | PASS | Cat | 0.9922120222098424 | ratio | Meets Cat threshold (margin +0.01221). |
| Sn_T2_0_M8 | T2 | M8 | FAIL |  | 0.5697834915886127 | abs_ratio | Violates Maj threshold by 0.5198 (above required value). Apply regularisation or data augmentation to shrink the generalisation gap. |
| Sn_T3_0_M13 | T3 | M13 | PASS | Cat | 0.0001074655870405 | variance | Meets Cat threshold (margin +0.04989). Per-client accuracy range 0.9432-0.9794. |
| Sn_T3_0_M14 | T3 | M14 | PASS | Cat | 0.0103665610035632 | std | Meets Cat threshold (margin +0.03963). Per-client accuracy range 0.9432-0.9794. |
| Sn_T3_0_M3 | T3 | M3 | FAIL |  | 0.555045871559633 | ratio | Violates Maj threshold by 0.345 (below required value). Increase recall through threshold tuning or focused retraining on failure cases. |
| Sn_T3_0_M5 | T3 | M5 | FAIL |  | 16.113086700439453 | cycles | Violates Maj threshold by 6.405 (above required value). Improve RUL estimator accuracy or extend the proactive maintenance lead time. |
| Sn_T4_0_M3 | T4 | M3 | FAIL |  | 0.6055045871559633 | ratio | Violates Maj threshold by 0.2945 (below required value). Increase recall through threshold tuning or focused retraining on failure cases. |
| Sn_T4_0_M35 | T4 | M35 | PASS | Cat | 5.913993930545581e-05 | variance | Meets Cat threshold (margin +0.04994). Per-client accuracy range 0.9574-0.9784. |
| Sn_T4_0_M5 | T4 | M5 | FAIL |  | 16.45545768737793 | cycles | Violates Maj threshold by 6.747 (above required value). Improve RUL estimator accuracy or extend the proactive maintenance lead time. |
| Sn_T5_0_M10 | T5 | M10 | PASS | Cat | 50.0 | rounds | Meets Cat threshold (margin +0). |
| Sn_T5_0_M11 | T5 | M11 | PASS | Cat | 0.2666244506835937 | MB | Meets Cat threshold (margin +49.73). |
| Sn_T5_0_M12 | T5 | M12 | FAIL |  | 3.7179450323580143 | x | Violates Maj threshold by 1.282 (below required value). Optimise communication budget and scheduling (compression, partial participation). |
| Sn_T5_0_M30 | T5 | M30 | PASS | Cat | 250.0 | seconds | Meets Cat threshold (margin +3350). |
| Sn_T6_0_M21 | T6 | M21 | FAIL |  | 0.0 | ratio | Violates Maj threshold by 0.85 (below required value). Enable robust aggregation and investigate anomalous client behaviours. |
| Sn_T6_0_M22 | T6 | M22 | PASS | Cat | 0.992528115852719 | ratio | Meets Cat threshold (margin +0.07253). |
| Sn_T7_0_M20 | T7 | M20 | PASS | Haz | 5.0 | epsilon | Meets Haz threshold (margin +0). |
| Sn_T7_0_M3 | T7 | M3 | FAIL |  | 0.6697247706422018 | ratio | Violates Maj threshold by 0.2303 (below required value). Increase recall through threshold tuning or focused retraining on failure cases. |
| Sn_T8_0_M10 | T8 | M10 | PASS | Cat | 50.0 | rounds | Meets Cat threshold (margin +0). |
| Sn_T8_0_M36 | T8 | M36 | PASS | Haz | 0.8 | ratio | Meets Haz threshold (margin +0.05). |
| Sn_T9_0_M24 | T9 | M24 | ALARM | Cat | 0.0 | p_value | Alarm triggered at Cat (value 0 < alpha 0.01). KS statistic 0.3230. Trigger drift response plan and recalibrate with recent operational data. |
