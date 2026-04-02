DOMAIN_REGISTRY = {
    # 1. Retail & E-Commerce (B2C/Marketplace)
    "retail": {
        "keywords": ["gmv", "sku", "basket_size", "upt", "sell_through", "markdown", "repurchase", "cannibalization", "stockout"],
        "metrics": "Retail KPIs: \n"
                   "- RFM: Recency (Days since last order), Frequency (Order count), Monetary (LTV). \n"
                   "- UPT (Units Per Transaction): Total Units / Total Orders. \n"
                   "- Sell-Through Rate: Units Sold / (Starting Inventory + Receipts). \n"
                   "- Markdown Depth: (MSRP - Actual Price) / MSRP. \n"
                   "- Stock-to-Sales Ratio: Average Inventory Value / Net Sales.",
        "sql_tips": "1. Weighted Average: `SUM(revenue)/SUM(quantity)` to avoid price dilution. \n"
                    "2. Cohort Repurchase: `COUNT(DISTINCT user_id) WHERE order_count > 1`. \n"
                    "3. New/Old Flag: `MIN(order_date) OVER(PARTITION BY user_id)` logic."
    },

    # 2. Finance & Fintech (Banking/Lending)
    "finance": {
        "keywords": ["dpd", "apr", "amortization", "principal", "vintage", "fpd", "collateral", "nav", "disbursement"],
        "metrics": "Finance KPIs: \n"
                   "- DPD (Days Past Due): 30+, 60+, 90+ aging buckets. \n"
                   "- Bad Rate: Delinquent Principal / Total Outstanding Principal. \n"
                   "- FPD7 (First Payment Default): % of users defaulting on the very first installment. \n"
                   "- APR: Real annualized cost including fees. \n"
                   "- Vintage Loss: Cumulative loss rate grouped by disbursement month.",
        "sql_tips": "1. Aging Buckets: `CASE WHEN dpd BETWEEN 1 AND 30 THEN 'M1'`. \n"
                    "2. Cumulative Recovery: `SUM(repayment) OVER(PARTITION BY loan_id ORDER BY date)`. \n"
                    "3. Division Safety: `NULLIF(total_principal, 0)`."
    },

    # 3. SaaS & Subscriptions (B2B)
    "saas_subscription": {
        "keywords": ["mrr", "arr", "churn", "nrr", "cac", "ltv", "upsell", "downsell", "seat_utilization", "payback"],
        "metrics": "SaaS KPIs: \n"
                   "- MRR (Monthly Recurring Revenue): Normalized monthly subscription value. \n"
                   "- Net Revenue Retention (NRR): (Starting MRR + Expansion - Churn - Downsell) / Starting MRR. \n"
                   "- Churn Rate: Cancelled Customers / Total Customers at period start. \n"
                   "- LTV/CAC Ratio: Targeted > 3x for healthy growth. \n"
                   "- Logo Churn vs Revenue Churn.",
        "sql_tips": "1. MRR Movement: Compare `plan_id` between `T` and `T-1`. \n"
                    "2. Active Warning: Zero `login_event` in the last 7 days. \n"
                    "3. Revenue Recognition: Flatten annual prepayments into 12 monthly rows."
    },

    # 4. Manufacturing & Industry 4.0
    "manufacturing": {
        "keywords": ["oee", "yield", "downtime", "mtbf", "takt_time", "spc", "bom", "wip", "cycle_time"],
        "metrics": "Industrial KPIs: \n"
                   "- OEE (Overall Equipment Effectiveness): Availability × Performance × Quality. \n"
                   "- First Pass Yield (FPY): Units passing first test / Total units started. \n"
                   "- MTBF (Mean Time Between Failures): Total uptime / Number of failures. \n"
                   "- Takt Time: Available Production Time / Customer Demand. \n"
                   "- Scrap Rate: Defective units / Total units produced.",
        "sql_tips": "1. Downtime Analysis: `timestamp - LAG(timestamp)` where `status='STOP'`. \n"
                    "2. SPC Control: `WHERE ABS(val - AVG(val) OVER()) > 3 * STDDEV(val) OVER()`. \n"
                    "3. BOM Roll-up: Recursive CTE for multi-level assemblies."
    },

    # 5. Healthcare & Clinical Operations
    "healthcare": {
        "keywords": ["patient", "icd-10", "mortality", "readmission", "los", "triage", "prescriptions", "insurance"],
        "metrics": "Healthcare KPIs: \n"
                   "- LOS (Length of Stay): Days between admission and discharge. \n"
                   "- Readmission Rate: % of patients returning within 30 days for the same diagnosis. \n"
                   "- Drug-to-Revenue Ratio: Pharmacy sales / Total hospital revenue. \n"
                   "- Bed Occupancy Rate: Occupied bed-days / Available bed-days. \n"
                   "- PMPM: Per Member Per Month cost for insurance.",
        "sql_tips": "1. PII Masking: `CONCAT(LEFT(name, 1), '***')`. \n"
                    "2. Patient Path: `visit -> diagnosis -> treatment -> billing`. \n"
                    "3. Waiting Time: `AVG(DATEDIFF(MINUTE, registration_ts, triage_ts))`."
    },

    # 6. Logistics & Last-Mile Delivery
    "logistics": {
        "keywords": ["last_mile", "otd", "freight", "sorting", "tracking", "dim_weight", "cod", "route_optimization"],
        "metrics": "Logistics KPIs: \n"
                   "- OTD (On-Time Delivery): Delivered on-time / Total orders. \n"
                   "- CPK (Cost Per Kilometer): Total transport cost / Total distance. \n"
                   "- Sortation Throughput: Parcels processed per hour. \n"
                   "- Dimensional Weight Ratio: Volumetric weight vs Actual weight. \n"
                   "- Damaged Rate: Reported damage / Total shipments.",
        "sql_tips": "1. Geolocation: `ST_Distance_Sphere` for Haversine distance. \n"
                    "2. Lead Time: `DATEDIFF(SECOND, pickup_time, delivery_time)`. \n"
                    "3. Stagnation: Find parcels with no `status_update` for > 24 hours."
    },

    # 7. EdTech & Learning Analytics
    "education": {
        "keywords": ["enrollment", "burned_lessons", "completion", "renewal", "k12", "quiz_score", "learning_days"],
        "metrics": "EdTech KPIs: \n"
                   "- Completion Rate: Students finishing 100% of modules / Total enrolled. \n"
                   "- Renewal/Retention: % of students buying the next level course. \n"
                   "- Burn Rate: Revenue recognized based on lessons actually consumed. \n"
                   "- LAD (Learning Active Days): Monthly days spent learning > 15 mins. \n"
                   "- Fill Rate: Enrolled students / Class capacity.",
        "sql_tips": "1. Student Ranking: `DENSE_RANK() OVER(ORDER BY score DESC)`. \n"
                    "2. Absenteeism: Identify missing records in `attendance` via `LEFT JOIN`. \n"
                    "3. Progress: `lessons_finished / total_lessons`."
    },

    # 8. AdTech & Digital Marketing
    "ad_tech": {
        "keywords": ["roas", "attribution", "cpm", "cpc", "ctr", "vtr", "conversion", "pixel", "bid_win_rate"],
        "metrics": "AdTech KPIs: \n"
                   "- ROAS (Return on Ad Spend): Revenue / Ad Spend. \n"
                   "- eCPM (Effective CPM): Total Earnings / (Impressions / 1000). \n"
                   "- Last-Click Attribution: Credit revenue to the final `click_id` before conversion. \n"
                   "- VTR (View-Through Rate): Complete video views / Total impressions. \n"
                   "- Bid Win Rate: Bids won / Total bids submitted.",
        "sql_tips": "1. Attribution Window: `JOIN clicks TO conversions` where `diff < 7 days`. \n"
                    "2. ROI Calc: `(Revenue - Spend) / Spend`. \n"
                    "3. Funnel: Impression -> Click -> Landing -> Purchase."
    },

    # 9. Energy & Grid Utilities
    "energy": {
        "keywords": ["peak_load", "curtailment", "lcoe", "renewable", "grid_loss", "carbon_intensity", "tariff"],
        "metrics": "Utility KPIs: \n"
                   "- Line Loss Rate: (Power Sent - Power Metered) / Power Sent. \n"
                   "- Curtailment: Potential wind/solar energy not harvested due to grid limits. \n"
                   "- Peak-to-Valley Ratio: Maximum Load / Minimum Load. \n"
                   "- Carbon Intensity: Grams of CO2 per kWh produced. \n"
                   "- Load Factor: Average Load / Peak Load over a period.",
        "sql_tips": "1. Time-of-Use: `CASE WHEN HOUR(ts) BETWEEN 18 AND 22 THEN 'Peak'`. \n"
                    "2. Interval Aggregation: `GROUP BY (UNIX_TIMESTAMP(ts) DIV 900)` for 15-min bars. \n"
                    "3. Spike Detection: `val > 1.2 * LAG(val)`."
    },

    # 10. Real Estate & PropTech
    "real_estate": {
        "keywords": ["vacancy", "revpar", "adr", "absorption", "cap_rate", "noi", "sqft", "listing_aging"],
        "metrics": "PropTech KPIs: \n"
                   "- RevPAR (Revenue Per Available Room): Occupancy × ADR. \n"
                   "- Absorption Period: Total Inventory / Monthly Sales Velocity. \n"
                   "- Rent-to-Price Ratio: Annual Rent / Property Value. \n"
                   "- NOI (Net Operating Income): Gross Income - Operating Expenses. \n"
                   "- Vacancy Rate: Unoccupied Units / Total Portfolio.",
        "sql_tips": "1. Inventory Age: `DATEDIFF(NOW(), listed_date)`. \n"
                    "2. Comp-Analysis: `AVG(price) OVER(PARTITION BY zip_code)`. \n"
                    "3. Yield: `(rent * 12) / purchase_price`."
    },

    # 11. Gaming (Mobile & F2P)
    "gaming": {
        "keywords": ["whale", "gacha", "arpdau", "battle_pass", "tutorial_completion", "sink_source", "retention"],
        "metrics": "Gaming KPIs: \n"
                   "- ARPDAU: Average Revenue Per Daily Active User. \n"
                   "- Whale Concentration: % of total revenue from the top 1% of spenders. \n"
                   "- Source/Sink Ratio: Virtual currency produced / Virtual currency consumed. \n"
                   "- Pity Rate: Probability adjustment in Gacha after N consecutive losses. \n"
                   "- D1/D7/D30 Retention.",
        "sql_tips": "1. Economy Audit: `SUM(earn) - SUM(spend)` grouped by player. \n"
                    "2. Drop-off: Identify max level reached before 7-day inactivity. \n"
                    "3. Session Length: `MAX(ts) - MIN(ts)` per login session."
    },

    # 12. GovTech & Public Services
    "government": {
        "keywords": ["compliance", "budget_execution", "citizen_feedback", "tat", "welfare", "transparency"],
        "metrics": "GovTech KPIs: \n"
                   "- Case Closure Rate: Resolved applications / Total applications. \n"
                   "- TAT (Turnaround Time): Business days from filing to resolution. \n"
                   "- Budget Execution: Actual expenditure / Allocated budget. \n"
                   "- Service Reach: Citizens served / Total eligible population. \n"
                   "- PII Accuracy: Audit of data completeness.",
        "sql_tips": "1. Business Day Lead Time: Exclude weekends via `DAYOFWEEK`. \n"
                    "2. Recursive Dept: `WITH RECURSIVE` to traverse org charts. \n"
                    "3. Masking: Enforce `SHA2()` for personal identifiers."
    },

    # 13. HR Analytics & Talent Management
    "hr_analytics": {
        "keywords": ["attrition", "compa-ratio", "headcount", "recruitment_funnel", "tenure", "payroll_flux"],
        "metrics": "HR KPIs: \n"
                   "- Attrition Rate: Terminated employees / Average headcount. \n"
                   "- Compa-Ratio: Actual Salary / Midpoint of Salary Range. \n"
                   "- Revenue per Employee: Total Revenue / Total FTE. \n"
                   "- Recruitment Conversion: Offer Accepted / Total Interviews. \n"
                   "- Average Tenure: Total months of service across active employees.",
        "sql_tips": "1. Attrition Cohort: Group by `hire_quarter` and `department`. \n"
                    "2. Seniority: `DATEDIFF(NOW(), hire_date) / 365`. \n"
                    "3. Pipeline Speed: Time diff between recruitment status changes."
    },

    # 14. Cloud Infrastructure & Data Centers
    "it_infrastructure": {
        "keywords": ["pue", "uptime", "latency", "provisioning", "iops", "bandwidth", "sla", "vcpus"],
        "metrics": "IT KPIs: \n"
                   "- PUE (Power Usage Effectiveness): Total Facility Power / IT Equipment Power. \n"
                   "- Availability (Uptime %): (Total time - Downtime) / Total time. \n"
                   "- P99 Latency: 99th percentile of response times. \n"
                   "- IOPS: Input/Output operations per second. \n"
                   "- Cost per vCPU: Infrastructure cost / Allocated vCPUs.",
        "sql_tips": "1. Utilization P95: `PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_usage)`. \n"
                    "2. Under-utilization: Find instances with `AVG(cpu) < 5%` for 7 days. \n"
                    "3. Error Rate: `count(errors) / total_requests`."
    },

    # 15. Cross-Border Trade & Customs
    "cross_border": {
        "keywords": ["hs_code", "clearance", "bonded", "incoterms", "duty", "port_congestion", "tariff"],
        "metrics": "Trade KPIs: \n"
                   "- Clearance TAT: Declaration time to Release time. \n"
                   "- Inspection Rate: % of shipments physically checked by customs. \n"
                   "- Duty-to-Value Ratio: Total duties paid / Total declared value. \n"
                   "- Bonded Turnover: Average days goods stay in a bonded warehouse. \n"
                   "- Compliance Error Rate: % of filings requiring correction.",
        "sql_tips": "1. Tax Simulation: `declared_val * hs_rate`. \n"
                    "2. Congestion: `AVG(wait_days)` grouped by `port_id`. \n"
                    "3. FX Audit: `amount * (SELECT rate FROM fx WHERE date = trx_date)`."
    },

    # 16. F&B & Instant Delivery (Q-Commerce)
    "food_delivery": {
        "keywords": ["prep_time", "spillage", "courier_subsidy", "radius", "rating", "table_turnover"],
        "metrics": "F&B KPIs: \n"
                   "- Prep Time: Merchant接单 to 出餐时耗. \n"
                   "- Delivery Radius Penetration: Orders grouped by distance buckets (0-1km, 1-3km). \n"
                   "- Courier Subsidy Ratio: Delivery subsidies / Total delivery revenue. \n"
                   "- Table Turnover: (Dine-in) Total guests / Total tables / Hours. \n"
                   "- Order Accuracy: Orders without 'missing item' complaints / Total orders.",
        "sql_tips": "1. Late Flag: `actual_arrival > promised_arrival`. \n"
                    "2. Spatial Density: `COUNT(orders) GROUP BY ST_GeoHash(location)`. \n"
                    "3. Top Dishes: `ORDER BY count DESC LIMIT 3` per restaurant."
    },

    # 17. Deep Gaming Economy (Numerical Balancing)
    "gaming_economy": {
        "keywords": ["inflation", "loot_box", "sink_efficiency", "soft_cap", "currency_velocity"],
        "metrics": "Economy KPIs: \n"
                   "- Currency Inflation Rate: Month-over-month growth of average player balance. \n"
                   "- Pity Timer Success: Avg pulls needed to trigger guaranteed rare items. \n"
                   "- Sink Efficiency: Total currency removed via mechanics / Total currency produced. \n"
                   "- High-Value Player Churn Warning: Spend drop > 50% for top 1% spenders.",
        "sql_tips": "1. Balance Ledger: `SUM(source) - SUM(sink)`. \n"
                    "2. Batch Gacha: Cluster records with identical `user_id` and `timestamp`. \n"
                    "3. Purchasing Power: `item_price / avg_player_daily_income`."
    },

    # 18. Customer Success (B2B SaaS)
    "saas_success": {
        "keywords": ["nps", "health_score", "onboarding_tat", "expansion_mrr", "ticket_resolution"],
        "metrics": "CS KPIs: \n"
                   "- Customer Health Score: Weighted index of logins, feature usage, and support tickets. \n"
                   "- Time to Value (TTV): Days from signup to first 'Aha!' moment. \n"
                   "- Expansion MRR: Revenue growth from existing customers (Upsells). \n"
                   "- NPS (Net Promoter Score): (Promoters - Detractors) / Total respondents.",
        "sql_tips": "1. Risk Alert: `MAX(last_activity) < TODAY - 14`. \n"
                    "2. Ticket Aging: `CURRENT_DATE - created_at` for open tickets. \n"
                    "3. Feature Adoption: `COUNT(DISTINCT feature_id) / Total_Features`."
    },

    # 19. Legal, Risk & Compliance
    "legal_compliance": {
        "keywords": ["litigation", "kyc", "aml", "clause_audit", "validity", "exposure_amount"],
        "metrics": "Legal KPIs: \n"
                   "- Contract Cycle Time: Draft to Final Signature. \n"
                   "- KYC Failure Rate: % of users failing identity verification. \n"
                   "- Litigation Success Rate: Cases won or settled favorably / Total cases. \n"
                   "- Total Exposure: Sum of potential liabilities in active lawsuits. \n"
                   "- Compliance Pass Rate: Successful audits / Total audits.",
        "sql_tips": "1. Renewal Alert: `expiry_date - 30 days`. \n"
                    "2. Search Audit: `WHERE contract_text LIKE '%liability_limit%'`. \n"
                    "3. Version Control: `MAX(version_no) OVER(PARTITION BY doc_id)`."
    },

    # 20. ESG & Sustainability (Green Energy)
    "esg_sustainability": {
        "keywords": ["carbon_footprint", "emission_intensity", "waste_recovery", "renewable_ratio", "diversity_ratio"],
        "metrics": "ESG KPIs: \n"
                   "- Carbon Intensity: Metric tons of CO2 per $1M revenue. \n"
                   "- Renewable Energy Ratio: Green energy consumed / Total energy consumed. \n"
                   "- Diversity Ratio: Gender/Ethnic representation in leadership. \n"
                   "- Waste Diversion: Waste recycled / Total waste produced. \n"
                   "- Safety Frequency (TRIR): Total recordable incidents per 200,000 man-hours.",
        "sql_tips": "1. Emission Trend: `total_emissions / total_output` monthly. \n"
                    "2. Diversity: `COUNT(minority) / COUNT(*)`. \n"
                    "3. Energy Shift: Compare current green % vs last year."
    },

    # 21. Agri-Tech (Precision Farming)
    "agri_tech": {
        "keywords": ["yield_per_acre", "soil_moisture", "irrigation_efficiency", "pest_damage", "harvest_loss"],
        "metrics": "Agri KPIs: \n"
                   "- Yield per Acre: Harvest weight / Total planted area. \n"
                   "- Irrigation Efficiency: Crop water intake / Total water applied. \n"
                   "- Fertilizer ROI: Increase in yield value / Cost of fertilizer. \n"
                   "- Pest Damage Rate: Area impacted by pests / Total area. \n"
                   "- Harvest Loss Ratio: Weight lost during picking and transport.",
        "sql_tips": "1. Plot Aggregation: `AVG(moisture)` grouped by `plot_id`. \n"
                    "2. Season Comparison: Current yield vs 3-year historical average. \n"
                    "3. Sensor Health: Identify sensors with zero variance in readings."
    },

    # 22. Automotive & Telematics (Connected Cars)
    "automotive": {
        "keywords": ["vin", "battery_soh", "dtc_frequency", "mileage", "adas_engagement", "charging_cycle"],
        "metrics": "Auto KPIs: \n"
                   "- Battery SOH (State of Health): Current capacity / Original capacity. \n"
                   "- DTC Frequency: Number of Diagnostic Trouble Codes reported per 1,000 km. \n"
                   "- ADAS Engagement: % of mileage driven with assistance features active. \n"
                   "- Average Range Achievement: Actual range / Estimated range. \n"
                   "- Energy Efficiency: kWh per 100 km.",
        "sql_tips": "1. Fleet Mileage: `MAX(mileage) - MIN(mileage)` by `vin`. \n"
                    "2. Charging Behavior: `soc_end - soc_start`. \n"
                    "3. VIN Maintenance: Join `vehicle_telemetry` with `service_records`."
    },

    # 23. Media & Creator Economy
    "creator_economy": {
        "keywords": ["paywall_conversion", "recirculation", "royalty_split", "content_half-life", "arpu"],
        "metrics": "Media KPIs: \n"
                   "- Content Half-Life: Time taken for a post to reach 50% of its lifetime views. \n"
                   "- Paywall Conversion: % of readers subscribing after hitting a paywall. \n"
                   "- Recirculation Rate: % of users clicking another internal link after reading an article. \n"
                   "- Royalty Pool Distribution: (Author Views / Total Views) × Total Payout Pool. \n"
                   "- Subscriber Churn Rate.",
        "sql_tips": "1. Evergreen Content: Posts with constant traffic > 30 days. \n"
                    "2. User Value: `SUM(subscription_fee + ad_revenue) / count(users)`. \n"
                    "3. Engagement Depth: `AVG(scroll_percentage)`."
    },

    # 24. Metaverse & Web3 (Virtual Assets)
    "metaverse_nft": {
        "keywords": ["floor_price", "gas_fee_ratio", "holder_concentration", "mint_success", "secondary_velocity"],
        "metrics": "Web3 KPIs: \n"
                   "- Floor Price Stability: Standard deviation of the minimum listing price. \n"
                   "- Secondary Market Velocity: Volume of secondary sales / Total supply. \n"
                   "- Holder Concentration: % of assets held by the top 10 wallets (Gini coefficient). \n"
                   "- Gas Fee Ratio: Transaction fees / Total transaction value. \n"
                   "- Mint Success Rate: Completed mints / Total mint attempts.",
        "sql_tips": "1. Floor Detection: `MIN(listing_price) WHERE status='listed'`. \n"
                    "2. Unique Holders: `COUNT(DISTINCT wallet_address)`. \n"
                    "3. Royalty Calc: `total_sales * royalty_percentage`."
    },

    # 25. Travel & Luxury Hospitality
    "travel_hotel": {
        "keywords": ["revpar", "adr", "occupancy", "ota_commission", "booking_window", "cancellation_lead_time"],
        "metrics": "Hospitality KPIs: \n"
                   "- RevPAR (Revenue Per Available Room): ADR × Occupancy %. \n"
                   "- ADR (Average Daily Rate): Room Revenue / Rooms Sold. \n"
                   "- Booking Window: Days between booking and check-in. \n"
                   "- OTA Commission Ratio: Commission paid to platforms / Total revenue. \n"
                   "- Cancellation Lead Time: Days before check-in that cancellations occur.",
        "sql_tips": "1. RevPAR: `SUM(room_revenue) / SUM(available_rooms)`. \n"
                    "2. Channel Mix: `GROUP BY source_channel` (Direct vs OTA). \n"
                    "3. Occupancy Curve: Daily `sum(rooms_sold)` vs total capacity."
    },

    # 26. Cybersecurity & Threat Intel
    "cybersecurity": {
        "keywords": ["mttd", "mttr", "false_positive_rate", "vulnerability_score", "phishing_click_rate"],
        "metrics": "Security KPIs: \n"
                   "- MTTD (Mean Time to Detect): Time from intrusion to detection. \n"
                   "- MTTR (Mean Time to Respond): Time from detection to remediation. \n"
                   "- False Positive Rate: Alerts wrongly identified as threats / Total alerts. \n"
                   "- Phishing Click Rate: % of employees clicking simulated malicious links. \n"
                   "- Vulnerability Patch Time: Days between patch release and implementation.",
        "sql_tips": "1. Attack Pattern: `GROUP BY source_ip HAVING COUNT > threshold`. \n"
                    "2. Latency: `detected_at - intrusion_at`. \n"
                    "3. Risk Level: `COUNT(*)` grouped by `severity`."
    },

    # 27. BioTech & Pharma R&D
    "biotech": {
        "keywords": ["clinical_trial_phase", "efficacy_rate", "sample_purity", "molecule_affinity", "stability_index"],
        "metrics": "BioTech KPIs: \n"
                   "- Trial Success Rate: % of molecules moving from Phase I to Phase II. \n"
                   "- Efficacy Ratio: Treatment group outcome vs Control group outcome. \n"
                   "- Sample Purity: Main compound concentration / Total sample volume. \n"
                   "- Stability Index: Maintenance of compound properties over time (shelf-life). \n"
                   "- R&D Yield: Successful discoveries / Total experiments conducted.",
        "sql_tips": "1. Trial Results: `AVG(outcome)` grouped by `dosage_level`. \n"
                    "2. Correlation: `CORR(dose, response)`. \n"
                    "3. Failure Analysis: `GROUP BY error_code` in lab logs."
    },

    # 28. Aviation & Fleet Ops
    "aviation": {
        "keywords": ["casm", "pask", "load_factor", "on-time_performance", "fuel_burn_rate", "turnaround_time"],
        "metrics": "Aviation KPIs: \n"
                   "- CASM (Cost per Available Seat Mile): Total operating cost / ASMs. \n"
                   "- Load Factor: Occupied seats / Available seats. \n"
                   "- Turnaround Time (TAT): Time from landing to next takeoff. \n"
                   "- Fuel Burn Rate: Gallons of fuel per flight hour. \n"
                   "- On-Time Performance (OTP): % of flights arriving within 15 mins of schedule.",
        "sql_tips": "1. Fuel Efficiency: `SUM(fuel) / SUM(flight_hours)`. \n"
                    "2. Delay Breakdown: `SUM(delay_minutes) GROUP BY cause_code`. \n"
                    "3. Fleet Age: `DATEDIFF(NOW(), delivery_date)` per aircraft."
    }
}