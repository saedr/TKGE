# FactGate causal-fact sparsity pilot

## Decision: **INCONCLUSIVE_WEAK_PROBE**

Frozen question: among consequential write decisions that the local probe reproduces at baseline, is action sensitivity concentrated in a very small subset of structured tool-return facts?

## Primary results

| Quantity | Value |
|---|---:|
| Candidate consequential decisions | 54 |
| Baseline-correct tool decisions | 15 |
| Baseline tool accuracy | 0.278 |
| Sensitive decisions | 6 |
| Sensitive / baseline-correct | 0.400 |
| Median fraction of facts needed for 80% of influence | 0.500 |
| Mean fraction of facts needed for 80% of influence | 0.458 |
| Sensitive decisions individually <=20% | 0.167 |
| Median total influence | 6.000 |
| Median max single-fact influence | 1.000 |

## Frozen gate

CONTINUE only if baseline-correct decisions >= 20, sensitive decisions >= 10, and median 80%-coverage fact fraction <= 0.20.

## Interpretation guardrail

This is a phenomenon pilot on historical tau-bench airline states. It tests sparsity of decision dependence under same-field factual counterfactuals. It is not a final agent-safety benchmark and does not claim that the historical trajectories themselves are current tau3-bench results.

## Most influential repeated fields (descriptive)

| Field | n | Mean influence | Nonzero rate |
|---|---:|---:|---:|
| search_direct_flight.flight_number | 4 | 0.500 | 0.500 |
| get_user_details.address.address1 | 2 | 0.500 | 0.500 |
| get_user_details.address.province | 2 | 0.500 | 0.500 |
| get_user_details.address.zip | 2 | 0.500 | 0.500 |
| get_user_details.name.first_name | 2 | 0.500 | 0.500 |
| get_user_details.saved_passengers.dob | 2 | 0.500 | 0.500 |
| search_direct_flight.available_seats.basic_economy | 2 | 0.500 | 0.500 |
| search_direct_flight.available_seats.business | 2 | 0.500 | 1.000 |
| search_direct_flight.available_seats.economy | 2 | 0.500 | 0.500 |
| search_direct_flight.destination | 2 | 0.500 | 0.500 |
