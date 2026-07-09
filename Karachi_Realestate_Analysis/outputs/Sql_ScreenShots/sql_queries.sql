-- =============================================================================
-- Project  : Karachi Real Estate Analysis
-- Database : karachi_realestate
-- File     : sql_queries.sql
-- Description: All analytical queries. Views must be created first (Section 1)
--              before running any query in Section 2+.
-- =============================================================================

USE karachi_realestate;


-- =============================================================================
-- SECTION 1 — BASE VIEWS  (run once after table is loaded)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- View 1: vw_reliable_areas
-- Purpose : Areas with ≥200 listings, each annotated with their median
--           price/sqft and market tier. Used as the foundation for all
--           downstream queries to avoid repeating the same CTE block.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_reliable_areas AS
WITH ranked AS (
    SELECT
        location,
        price_per_sqft,
        market_tier,
        ROW_NUMBER() OVER (PARTITION BY location ORDER BY price_per_sqft) AS rn,
        COUNT(*)     OVER (PARTITION BY location)                          AS listing_count
    FROM featured_engineering_listings
),
area_stats AS (
    SELECT
        location,
        listing_count,
        AVG(price_per_sqft)  AS median_ppsqft,
        MAX(market_tier)     AS market_tier
    FROM ranked
    WHERE rn IN (
        FLOOR((listing_count + 1) / 2),
        FLOOR((listing_count + 2) / 2)
    )
    GROUP BY location, listing_count
)
SELECT *
FROM area_stats
WHERE listing_count >= 200;


-- -----------------------------------------------------------------------------
-- View 2: listings_with_deals
-- Purpose : Every listing in a reliable area, enriched with its area's median
--           price/sqft and each listing's % deviation from that median.
--           Positive pct_vs_median  → overpriced vs neighbourhood.
--           Negative pct_vs_median  → potential deal vs neighbourhood.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW listings_with_deals AS
WITH area_medians AS (
    SELECT
        location,
        AVG(price_per_sqft) AS median_ppsqft
    FROM (
        SELECT
            location,
            price_per_sqft,
            ROW_NUMBER() OVER (PARTITION BY location ORDER BY price_per_sqft) AS rn,
            COUNT(*)     OVER (PARTITION BY location)                          AS cnt
        FROM featured_engineering_listings
    ) ranked
    WHERE rn IN (
        FLOOR((cnt + 1) / 2),
        FLOOR((cnt + 2) / 2)
    )
    GROUP BY location
)
SELECT
    o.*,
    m.median_ppsqft,
    ((o.price_per_sqft - m.median_ppsqft) / m.median_ppsqft) * 100 AS pct_vs_median
FROM featured_engineering_listings o
JOIN area_medians      m ON o.location = m.location
JOIN vw_reliable_areas r ON o.location = r.location;


-- =============================================================================
-- SECTION 2 — ANALYTICAL QUERIES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Query 1: Market Tier Distribution
-- Purpose : Count of listings per market tier across the entire dataset.
-- -----------------------------------------------------------------------------
SELECT
    market_tier,
    COUNT(*) AS count
FROM featured_engineering_listings
GROUP BY market_tier
ORDER BY count DESC;


-- -----------------------------------------------------------------------------
-- Query 2: Reliable Areas — Median Price/Sqft Ranking
-- Purpose : Rank areas with ≥200 listings by their median price per sqft.
--           Gives a statistically stable picture of area-level pricing.
-- -----------------------------------------------------------------------------
SELECT
    location,
    ROUND(median_ppsqft, 2) AS median_ppsqft,
    listing_count,
    market_tier
FROM vw_reliable_areas
ORDER BY median_ppsqft DESC;


-- -----------------------------------------------------------------------------
-- Query 3: Bedroom-Level Median Prices in the Top-5 Most Expensive Areas
-- Purpose : For each bedroom count (min 10 listings), show the median price
--           so buyers can benchmark against area norms.
-- -----------------------------------------------------------------------------
WITH top5 AS (
    SELECT location
    FROM vw_reliable_areas
    ORDER BY median_ppsqft DESC
    LIMIT 5
),
bed_counts AS (
    SELECT
        o.location,
        o.bedrooms,
        COUNT(*) AS count
    FROM featured_engineering_listings o
    JOIN top5 t ON o.location = t.location
    GROUP BY o.location, o.bedrooms
    HAVING COUNT(*) >= 10
),
ranks AS (
    SELECT
        location,
        bedrooms,
        price,
        ROW_NUMBER() OVER (PARTITION BY location, bedrooms ORDER BY price) AS rn,
        COUNT(*)     OVER (PARTITION BY location, bedrooms)                 AS grp_count
    FROM featured_engineering_listings
    WHERE location IN (SELECT location FROM top5)
),
bed_median AS (
    SELECT
        location,
        bedrooms,
        AVG(price) AS price
    FROM ranks
    WHERE rn IN (
        FLOOR((grp_count + 1) / 2),
        FLOOR((grp_count + 2) / 2)
    )
    GROUP BY location, bedrooms
)
SELECT
    bc.location,
    bc.bedrooms,
    bc.count,
    bm.price
FROM bed_counts bc
JOIN bed_median bm
    ON bc.location = bm.location
   AND bc.bedrooms  = bm.bedrooms
ORDER BY bc.location, bc.bedrooms;


-- -----------------------------------------------------------------------------
-- Query 4: Median Affordability Years per Area and Market Tier
-- Purpose : How many years of median salary does the median property in each
--           reliable area cost? Ranked worst-to-best affordability.
-- -----------------------------------------------------------------------------
WITH ranked_aff AS (
    SELECT
        o.location,
        o.market_tier,
        o.affordability_years,
        ROW_NUMBER() OVER (
            PARTITION BY o.location, o.market_tier
            ORDER BY o.affordability_years
        ) AS rn,
        COUNT(*) OVER (
            PARTITION BY o.location, o.market_tier
        ) AS grp_count
    FROM featured_engineering_listings o
    JOIN vw_reliable_areas r ON o.location = r.location
)
SELECT
    location,
    market_tier,
    ROUND(AVG(affordability_years), 4) AS affordability_years
FROM ranked_aff
WHERE rn IN (
    FLOOR((grp_count + 1) / 2),
    FLOOR((grp_count + 2) / 2)
)
GROUP BY location, market_tier
ORDER BY affordability_years DESC;


-- -----------------------------------------------------------------------------
-- Query 5: Deal Finder — Listings Below Area Median
-- Purpose : Pull all listings priced .
--         .
-- -----------------------------------------------------------------------------
SELECT*
FROM listings_with_deals;
