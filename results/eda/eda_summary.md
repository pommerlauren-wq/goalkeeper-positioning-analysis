# EDA Summary

## Dataset sizes

| Dataset | Rows | Columns |
|---|---|---|
| shots_master_df | 87,111 | 48 |
| freeze_master_df | 1,110,839 | 10 |

## Shot sub-type breakdown

| Sub-type | Count | % of total |
|---|---|---|
| Open Play | 81,551 | 93.6% |
| Free Kick | 4,221 | 4.8% |
| Penalty | 1,310 | 1.5% |
| Corner | 28 | 0.0% |
| Kick Off | 1 | 0.0% |

## Goal conversion rates by sub-type

| Sub-type | Shots | Goals | Conversion |
|---|---|---|---|
| Open Play | 81,551 | 8,411 | 10.3% |
| Free Kick | 4,221 | 276 | 6.5% |
| Penalty | 1,310 | 970 | 74.0% |
| Corner | 28 | 10 | 35.7% |
| Kick Off | 1 | 0 | 0.0% |

## xG distribution

- Available for: 87,111 / 87,111 shots
- Mean: 0.1063
- Median: 0.0548
- Max: 0.9951

## Freeze frame coverage

- Shots with freeze frames: 85,964 / 87,111 (98.7%)
- Visible players per shot: min=1, median=13, max=21

## Goalkeeper identifiability (open-play)

- Open-play shots: 81,551
- With freeze frame: 81,551 (100.0%)
- With identifiable opponent GK: 81,453 (99.9% of open-play, 99.9% of those with freeze frames)

## Final modeling sample

- **Sample size: 81,453**
- Goal conversion rate: 10.3%
- Unique matches: 3,433

## Recommendation

YES — 81,453 samples comfortably exceeds the 10,000 target. Proceed with CNN
training.
