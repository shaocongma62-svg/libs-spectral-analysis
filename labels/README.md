# Unified Label Table

`unified_wt.csv` is intentionally ignored by Git because it contains local
assay-derived measurements. The repository keeps only
`unified_wt.example.csv`, a header-only schema template.

## Purpose

The template documents the input contract expected by the quantitative scripts
without publishing real sample identifiers or compositions. It prevents a
silent column-order mismatch when an authorized collaborator reconstructs the
local label table.

## Required columns

| Column | Meaning |
| --- | --- |
| `sample_id` | Stable identifier matching the local loader output. |
| `dataset` | Dataset identifier, currently `A` or `B`. |
| `category` | Sample provenance/category, such as `industrial_dust` or `pure_reagent`. |
| `is_verified` | `True` only when the composition is traceable to an approved assay or reference certificate. |
| `Fe` ... `Cl` | Elemental mass fractions in wt%, after any documented oxide-to-element conversion. |

## Creating a local label file

1. Copy `unified_wt.example.csv` to `unified_wt.csv`.
2. Add one row per independent physical sample, not one row per laser shot.
3. Populate wt% values only from authorized, traceable assay or certificate
   data; record uncertain samples with `is_verified=False`.
4. Keep `unified_wt.csv` local. It is ignored by Git and should not be pushed
   without explicit data-owner approval.
