# Safety, limitations and pre-deployment checks — ThermADNet

**Scope:** Research code and machine-learning modeling relating to thermal conditions and anomaly detection.  
**Issued:** 2026-10-17  
**Status:** Risk guidance only. No claim is made that any checklist item is already implemented, tested, legally approved or certified.

This document supplements [DISCLAIMER.md](DISCLAIMER.md) and the existing copyright/license terms without changing either. It does not prevent claims or waive mandatory legal duties.

## Risks to assess

- False negatives or false positives can occur when hardware, environment and telemetry differ from the training/evaluation data.
- Poorly calibrated thresholds may hide a real thermal hazard or disrupt operations.
- Anomaly scores are not automatically reliable physical safety measurements or certified alarms.

## Checks before live use

- [ ] Compare with calibrated sensors and independent physical protective controls before any field use.
- [ ] Document data provenance, units, latency, uncertainty, missed detections and test conditions.
- [ ] Validate the model against actual distribution shifts, missing sensors and adversarial or faulty data.
- [ ] Maintain monitoring, alarm escalation and manual override independent of the model.

## User and maintainer responsibilities

Begin in an isolated, authorized test environment; apply least privilege, backups, monitoring and an independent review of any high-impact outputs or actions. Where a project depends on third-party providers or real equipment, comply with vendor terms and safety instructions. Report suspected security flaws privately. Record actual test evidence, versions and unresolved risks rather than inferring that documentation is proof of implementation.

**Mandatory rights preserved:** Liability, product-safety, privacy, security and consumer obligations cannot be eliminated by this notice where applicable law forbids exclusion.
