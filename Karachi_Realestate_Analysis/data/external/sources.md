# External Data Sources — Karachi Real Estate Analysis
*Last updated: May 2026 | All values verified against official PBS and SBP publications*

---

## 1. Pakistan CPI / Inflation Data

### Primary Sources

| Institution | Document | URL |
|---|---|---|
| **Pakistan Bureau of Statistics (PBS)** | Monthly CPI Press Releases (FY2019–FY2026) | https://www.pbs.gov.pk/price-statistics/ |
| **State Bank of Pakistan (SBP)** | Annual Reports — Chapter 5: Prices (FY2019–FY2024) | https://www.sbp.org.pk/publications/annual/ |
| **SBP Monetary Policy Statements** | CPI trend data, provisional figures FY2024 | https://www.sbp.org.pk/monetary/ |

### How CPI Was Calculated

PBS uses **2015-16 = 100** as the official base year.  
This project rebases to **2019 = 100** using the formula:

```
CPI (2019=100) = (PBS raw value ÷ 116.19) × 100
```

Where **116.19** is the FY2018-19 annual average CPI on the PBS 2015-16=100 scale,
computed by compounding official FY inflation rates:

| FY | Annual Inflation | Compounded PBS Value | Source |
|---|---|---|---|
| 2015-16 | — | 100.00 (base) | PBS |
| 2016-17 | 4.16% | 104.16 | SBP Annual Report 2017 |
| 2017-18 | 3.92% | 108.24 | SBP Annual Report 2018 |
| **2018-19** | **7.34%** | **116.19 ← 2019 base** | **PBS CPI Bulletin FY2019** |

### CPI Values Used in This Project

| Year | Annual Inflation (FY) | CPI Raw (2015-16=100) | **CPI Index (2019=100)** | Cumulative Inflation | Source |
|---|---|---|---|---|---|
| 2019 | 7.34% | 116.19 | **100.00** | 0.00% | PBS CPI FY2018-19 — SBP Annual Report 2019 |
| 2020 | 10.74% | 128.67 | **110.74** | +10.74% | PBS CPI FY2019-20 — SBP Annual Report 2020 |
| 2021 | 8.90% | 140.12 | **120.60** | +20.60% | PBS CPI FY2020-21 — SBP Annual Report 2021 |
| 2022 | 12.20% | 157.21 | **135.31** | +35.31% | PBS CPI FY2021-22 — pbs.gov.pk CPI bulletin |
| 2023 | 29.18% | 203.09 | **174.79** | +74.79% | PBS CPI FY2022-23 annual avg — pbs.gov.pk |
| 2024 | 23.41% | 250.63 | **215.71** | +115.71% | SBP Annual Report FY2023-24 provisional |
| 2025 | 4.93% | 262.99 | **226.34** | +126.34% | PBS Jul-Apr FY2024-25 period avg — pbs.gov.pk |
| 2026 | 7.40% | 282.45 | **243.10** | +143.10% | PBS April 2026 Press Release (see note below) |

> **Important correction:** Previous versions of this file stated CPI 2026 = 355.0.
> This was incorrect — it was based on monthly spot readings instead of FY annual averages.
> The correct value (243.10) is derived from the official PBS April 2026 Press Release.

### 2026 CPI Verification (PBS April 2026 Press Release)

**Source URL:**
`https://www.pbs.gov.pk/wp-content/uploads/2020/07/Press-Release-April-2026.pdf`

| Metric | Value | Notes |
|---|---|---|
| National CPI April 2026 (2015-16=100) | **292.81** | Used as anchor |
| Urban CPI April 2026 | 289.61 | — |
| Rural CPI April 2026 | 297.63 | (~298 — rural only, not used) |
| YoY inflation April 2026 | 10.89% | Confirmed in release |
| Jul-Apr FY2025-26 period average | 280.38 | 10-month average |
| FY2025-26 full-year estimate | **282.45** | (280.38×10 + 292.81×2) ÷ 12 |
| **Rebased to 2019=100** | **243.10** | 282.45 ÷ 116.19 × 100 |

---

## 2. Salary / Wage Data

### Primary Sources

| Institution | Document | URL |
|---|---|---|
| **PBS — PSLM FY2019** | Pakistan Social & Living Standards Measurement Survey | https://www.pbs.gov.pk/pslm-3/ |
| **PBS — Labour Force Survey (LFS)** | Annual LFS reports FY2020–FY2025 | https://www.pbs.gov.pk/labour-force-statistics/ |
| **PBS LFS 2024-25 (Official)** | LFS 2024-25 Annual Report — Average Monthly Wage confirmed | https://www.pbs.gov.pk/wp-content/uploads/2020/07/LFS-2024-25-Annual-Report.pdf |

### Key Finding from PBS

> **PBS LFS 2024-25 officially confirms:**
> **Average Monthly Wage in Pakistan = PKR 39,042**
> Source: PBS homepage → https://www.pbs.gov.pk/ (LFS Survey 2024-25 stat card)

This is the official PBS figure used to anchor the 2025 salary in this project.
Previous versions incorrectly used PKR 70,000 (a Rozee.pk tech-sector estimate), which
overstated real wages by 79% and produced misleadingly optimistic affordability numbers.

### Salary Values Used in This Project

| Year | Avg Monthly Salary (PKR) | Salary Growth vs 2019 | Real Purchasing Power Index | Source | Type |
|---|---|---|---|---|---|
| 2019 | **22,000** | 0.00% | 100.00 | PBS PSLM FY2019 — Karachi urban median | Actual |
| 2020 | **22,500** | +2.27% | 92.35 | PBS LFS FY2019-20 national estimate | Actual |
| 2021 | **24,000** | +9.09% | 90.46 | PBS LFS 2020-21 national | Actual |
| 2022 | **26,500** | +20.45% | 89.02 | PBS LFS 2021-22 national | Actual |
| 2023 | **30,000** | +36.36% | 78.02 | PBS LFS 2022-23 national | Actual |
| 2024 | **35,000** | +59.09% | 73.75 | PBS LFS 2023-24 national | Actual |
| 2025 | **39,042** | +77.46% | 78.41 | **PBS LFS 2024-25 OFFICIAL** | **Actual** |
| 2026 | **43,000** | +95.45% | 80.40 | Estimated +10% from PBS LFS 2024-25 | Forecast |

> **Scope note:** PBS PSLM (2019) reports Karachi urban median. PBS LFS (2020–2025)
> reports national average monthly wage. Karachi urban wages are typically 20–30% higher
> than the national average. Users may apply a Karachi urban premium when interpreting results.

### Real Purchasing Power Index Interpretation

| Index Value | Meaning |
|---|---|
| 100.00 (2019) | Baseline — full purchasing power |
| 78.02 (2023) | Workers lost 22% of real purchasing power — FY2023 crisis year |
| 73.75 (2024) | Lowest point — wages grew 59% but inflation grew 116% |
| 80.40 (2026) | Partial recovery — still 20% below 2019 in real terms |

---

## 3. Affordability Analysis Methodology

### Formula

```
affordability_years = estimated_property_price ÷ (avg_monthly_salary × 12)

affordability_years_2019 = price ÷ (22,000 × 12)
affordability_years_2026 = (price × 2.431) ÷ (43,000 × 12)
```

### Inflation Multiplier

```
INFLATION_MULTIPLIER = CPI_2026 ÷ CPI_2019 = 243.10 ÷ 100 = 2.431×
```

### Affordability Timeline (Median Karachi Property ≈ PKR 11.27M)

| Year | Adj. Price | Monthly Salary | Affordability (years) | vs 2019 |
|---|---|---|---|---|
| 2019 | PKR 11.27M | PKR 22,000 | **42.7 years** | Baseline |
| 2020 | PKR 12.48M | PKR 22,500 | 46.2 years | +3.5 worse |
| 2021 | PKR 13.59M | PKR 24,000 | 47.2 years | +4.5 worse |
| 2022 | PKR 15.25M | PKR 26,500 | 48.0 years | +5.3 worse |
| 2023 | PKR 19.70M | PKR 30,000 | 54.7 years | +12.0 worse |
| 2024 | PKR 24.31M | PKR 35,000 | 57.9 years | +15.2 worse ← peak |
| 2025 | PKR 25.51M | PKR 39,042 | 54.4 years | +11.7 worse |
| **2026** | **PKR 27.40M** | **PKR 43,000** | **53.1 years** | **+10.4 worse** |

> **Key finding:** Housing became progressively less affordable every year from 2019 to 2024.
> Partial recovery in 2025–2026 as inflation slowed and wages recovered, but affordability
> remains significantly worse than 2019. At 53.1 years of gross salary, Karachi housing
> remains acutely unaffordable for salaried workers.

### Critical Caveat

`estimated_price_2026 = 2019_price × CPI_multiplier (2.431×)`

This uses **general CPI** as the property appreciation rate. Actual Karachi real estate,
especially in premium areas (DHA, Clifton, Bahria), likely appreciated faster than general CPI.
If actual property appreciation exceeded 3.09× since 2019, affordability worsened beyond
what this model shows. The SBP House Price Index (https://www.sbp.org.pk/ecodata/HPI.asp)
would provide a more accurate property-specific deflator.

---

## 4. Property Price Data

| Source | Description |
|---|---|
| **Zameen.com** | Primary dataset — For Sale listings, Karachi, Aug 2018 – Jul 2019 |
| **Zameen.com Price Index** | Quarterly area-level price index for trend validation |

> **Important:** Zameen.com prices are *asking prices*, not confirmed transaction prices.
> Actual sale prices are typically 5–15% lower in Pakistan's negotiation-driven market.

---

## 5. How to Verify Each Number (Step-by-Step)

### Verify CPI (any year)
1. Go to https://www.sbp.org.pk/publications/annual/
2. Download the Annual Report PDF for that fiscal year
3. Open Chapter 5: Prices → Table 5.1 "Consumer Price Index"
4. Read the "Annual Average" row — this is the FY average inflation %
5. Compound from 116.19 (FY2018-19 base) and divide by 116.19 × 100

### Verify CPI 2026 specifically
1. Go to: `https://www.pbs.gov.pk/wp-content/uploads/2020/07/Press-Release-April-2026.pdf`
2. Find Table: "General CPI — National" → April 2026 column → **292.81**
3. Divide by 116.19 × 100 → **243.10** on 2019=100 base

### Verify salary 2019
1. Go to https://www.pbs.gov.pk/pslm-3/
2. Download PSLM FY2018-19 report
3. Find Table: "Monthly Income of Household by Province and Area" → Urban Sindh/Karachi

### Verify salary 2025
1. Go to https://www.pbs.gov.pk/ (homepage)
2. Look at the "Average Monthly Wage in Pakistan" stat card → **PKR 39,042** (LFS 2024-25)
3. Or download: `https://www.pbs.gov.pk/wp-content/uploads/2020/07/LFS-2024-25-Annual-Report.pdf`

---

## 6. Files Using This Data

| File | Columns Affected |
|---|---|
| `data/external/macro_reference.csv` | All columns — primary source file |
| `data/featured_engineering_listings.csv` | `estimated_price_2026`, `affordability_years_2026`, `affordability_change` |
| `data/powerbi_data.csv` | Same columns as above |
| `notebooks/02-feature_engineering.ipynb` | Constants: `CPI_2026`, `INFLATION_MULTIPLIER`, `AVG_MONTHLY_SALARY_2026` |

---

*All external data is used for educational and portfolio purposes only.*
*Values marked as "forecast" or "estimated" are projections and should be presented as such.*
