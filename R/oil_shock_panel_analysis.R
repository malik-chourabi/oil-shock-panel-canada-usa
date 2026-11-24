# =============================================================================
# PROJECT : Geopolitics, Oil Shock, and Macroeconomics — Panel Data Analysis
#           The Macroeconomic Impact of the 2026 Gulf Crisis
#           Countries : Canada & USA (Advanced Net-Exporters)
#           Period    : January 2023 – June 2026
#                       (April – June 2026 values are ARIMA-forecasted)
# =============================================================================
# AUTHORS  : Khadija Slaoui   (CPI Models + Diagnostics)
#            Jihen Cherif     (EDA, Descriptive Stats, IPI Model)
#            Emna Jaouadi     (Asymmetry Analysis, Country Heterogeneity, Scatter Plots)
#            Chayma Lessis    (Data Collection & ARIMA Forecasting)
#            Malik Chourabi   (Report Writing & Economic Interpretation)
# =============================================================================
# STRUCTURE OF THIS SCRIPT
#   PART 0  — Install & Load Libraries
#   PART 1  — Data Collection & Construction  [Step 2 NOTE]
#             1A. Raw data sources
#             1B. Log transformations (ln_CPI, ln_IPI, ln_NEER)
#             1C. Oil price percentage change
#             1D. ARIMA forecasting (April – June 2026)
#             1E. Asymmetric shock decomposition
#             1F. Final panel assembly
#   PART 2  — Load the Pre-built CSV and Inspect
#   PART 3  — Exploratory Data Analysis (EDA) & Descriptive Statistics
#             3A. Descriptive statistics table
#             3B. Plot: Brent crude oil price over time
#             3C. Plot: ln(CPI) for Canada & USA
#             3D. Plot: ln(IPI) for Canada & USA
#   PART 4  — Declare Panel Data Frame
#   PART 5  — CPI Inflation Models  [Member 1]
#             5A. Model 1 — Pooled OLS
#             5B. Model 2 — Fixed Effects (Within)
#             5C. Model 3 — Random Effects (Attempted via tryCatch — fails, N=2)
#   PART 6  — Diagnostic Tests  [Member 1]
#             6A. Breusch-Pagan LM Test (POLS vs panel)
#             6B. Hausman Test (Attempted via tryCatch — fails, RE=NULL)
#             6C. Autocorrelation Test (Breusch-Godfrey / Wooldridge)
#   PART 7  — Robust Standard Errors for CPI Models  [Member 1]
#             7A. Arellano HC3 robust SEs
#             7B. Stargazer regression table (POLS + FE side-by-side)
#   PART 8  — IPI Industrial Production Model  [Member 2]
#             8A. Fixed Effects model for ln(IPI)
#             8B. Arellano HC3 robust SEs
#             8C. Autocorrelation test — IPI
#             8D. Newey-West HAC robustness check
#             8E. Stargazer tables
#   PART 9  — Asymmetry Analysis  [Member 3]
#             9A. Asymmetric FE — ln(CPI)
#             9B. Asymmetric FE — ln(IPI)
#             9C. Stargazer asymmetry table
#   PART 10 — Heterogeneity Analysis  [Member 3 + Professor PDF]
#             10A. Oil Importers vs Exporters (Attempted via tryCatch — fails, all=exporters)
#             10B. Country-level: Canada — CPI and IPI
#             10C. Country-level: USA — CPI and IPI
#             10D. Stargazer comparison tables (Canada vs USA)
#   PART 11 — Scatter Plots  [Member 3]
#             11A. Oil shock vs ln(CPI)
#             11B. Oil shock vs ln(IPI)
#   PART 12 — Session Info (Reproducibility)
# =============================================================================


# =============================================================================
# Run from the repository root:  Rscript R/oil_shock_panel_analysis.R
# PART 0 — INSTALL AND LOAD LIBRARIES
# =============================================================================
# Uncomment and run the install line once on a fresh R installation:
# install.packages(c("tidyverse", "plm", "lmtest", "sandwich",
#                    "stargazer", "ggplot2", "car", "forecast"))

library(tidyverse)   # Data manipulation, pipes, ggplot2
library(plm)         # Panel data econometrics (POLS, FE, RE, tests)
library(lmtest)      # coeftest() — apply robust variance matrices
library(sandwich)    # vcovHC(), NeweyWest() — robust variance estimators
library(stargazer)   # Publication-quality regression tables
library(ggplot2)     # Advanced data visualisation
library(car)         # linearHypothesis() and other hypothesis tests
library(forecast)    # auto.arima() and forecast() for ARIMA projections

# Redirect all console output to a log file (split = TRUE keeps it on screen too)
dir.create("outputs", showWarnings = FALSE)
sink("outputs/GROUP_Results_Log.txt", split = TRUE)

cat("============================================================\n")
cat(" PROJECT: Geopolitics of Oil — 2026 Gulf Crisis\n")
cat(" Countries: Canada & USA | Period: Jan 2023 – Jun 2026\n")
cat("============================================================\n\n")


# =============================================================================
# PART 1 — DATA COLLECTION & CONSTRUCTION  [Step 2 NOTE]
# =============================================================================
# This section documents every transformation applied to the raw series
# before the final CSV (FINAL_panel_fixed.csv) was produced.
# The actual computation was performed once in R; the resulting CSV is then
# loaded in Part 2 for all subsequent analysis.
# =============================================================================

# -----------------------------------------------------------------------------
# 1A. RAW DATA SOURCES
# -----------------------------------------------------------------------------
# All series were downloaded in monthly frequency for Jan 2023 – Mar 2026
# (observed) and extended to Jun 2026 via ARIMA forecasting (see 1D).
#
#  Variable      Country     Source
#  ─────────────────────────────────────────────────────────────────────────
#  CPI           Canada      Statistics Canada (Table 18-10-0004-01)
#                USA         U.S. BLS (CPI-U, All Urban Consumers, FRED)
#  IPI           Canada      Statistics Canada (Table 16-10-0031-01)
#                USA         Federal Reserve / FRED (INDPRO)
#  NEER          Canada      Bank of Canada (CEER nominal effective rate)
#                USA         Federal Reserve / BIS (broad nominal index)
#  Policy Rate   Canada      Bank of Canada overnight rate target
#                USA         Federal Reserve federal funds rate (FRED)
#  Brent         Both        U.S. EIA / FRED (global benchmark, USD/barrel)
#  VIX           Both        CBOE Volatility Index via FRED (global variable)
#
# All indices were rebased to 2015 = 100 where necessary to ensure
# cross-country comparability before taking natural logarithms.
# -----------------------------------------------------------------------------

# ── Illustrative raw-data loading (replace paths with actual source files) ──

# raw_cpi_canada <- read.csv("raw_cpi_canada.csv")
# raw_cpi_usa    <- read.csv("raw_cpi_usa.csv")
# raw_ipi_canada <- read.csv("raw_ipi_canada.csv")
# raw_ipi_usa    <- read.csv("raw_ipi_usa.csv")
# raw_neer       <- read.csv("raw_neer.csv")       # Both countries
# raw_policy     <- read.csv("raw_policy_rate.csv") # Both countries
# raw_brent      <- read.csv("raw_brent.csv")       # One global series
# raw_vix        <- read.csv("raw_vix.csv")          # One global series


# -----------------------------------------------------------------------------
# 1B. LOG TRANSFORMATIONS
# -----------------------------------------------------------------------------
# Economic theory and panel data best-practice recommend working with natural
# logarithms of price / output indices. This:
#   (i)  compresses outliers and stabilises variance,
#  (ii)  makes coefficients interpretable as semi-elasticities,
# (iii)  renders the dependent variable closer to normally distributed.
#
# Three log variables are constructed:
#   ln_cpi  = log(CPI)   — dependent variable in the inflation equation
#   ln_ipi  = log(IPI)   — dependent variable in the growth equation
#   ln_neer = log(NEER)  — explanatory variable (exchange rate channel)
#
# Brent, VIX, and policy_rate are left in levels because:
#   • oil_pct_change (see 1C) already normalises oil into growth-rate form,
#   • VIX is already an index in natural units,
#   • policy_rate is a percentage — percentage change is not meaningful here.
# -----------------------------------------------------------------------------

# data_processed <- data_raw %>%
#   mutate(
#     ln_cpi  = log(cpi),    # Natural log of Consumer Price Index
#     ln_ipi  = log(ipi),    # Natural log of Industrial Production Index
#     ln_neer = log(neer)    # Natural log of Nominal Effective Exchange Rate
#   )


# -----------------------------------------------------------------------------
# 1C. OIL PRICE PERCENTAGE CHANGE
# -----------------------------------------------------------------------------
# Instead of using the Brent price level, we compute the monthly percentage
# change. This measures the oil *shock* rather than the absolute price level,
# which is the economically relevant driver of short-run inflation and output.
#
# Formula:
#   oil_pct_change_t = ( Brent_t − Brent_{t-1} ) / Brent_{t-1}  × 100
#
# Calculation is grouped by country so that the first observation within each
# country's time series (January 2023) returns NA (no prior month available).
# These NAs are set to 0 before modelling (one observation per country is lost
# under the alternative of dropping; setting to 0 preserves the panel balance).
#
# Two asymmetric shock columns are derived (used in Part 9):
#   oil_shock_pos = max(oil_pct_change, 0)   — captures only price INCREASES
#   oil_shock_neg = min(oil_pct_change, 0)   — captures only price DECREASES
# -----------------------------------------------------------------------------

# data_processed <- data_processed %>%
#   group_by(country) %>%
#   mutate(
#     oil_pct_change = (brent - lag(brent)) / lag(brent) * 100,
#     oil_pct_change = ifelse(is.na(oil_pct_change), 0, oil_pct_change),
#     oil_shock_pos  = ifelse(oil_pct_change > 0, oil_pct_change, 0),
#     oil_shock_neg  = ifelse(oil_pct_change < 0, oil_pct_change, 0)
#   ) %>%
#   ungroup()


# -----------------------------------------------------------------------------
# 1D. ARIMA FORECASTING — APRIL to JUNE 2026
# -----------------------------------------------------------------------------
# The observed data ends in March 2026. Because the research design requires
# capturing the post-crisis economic dynamics through at least one quarter,
# three additional months (April, May, June 2026) are forecasted per variable
# per country using ARIMA models estimated on the Jan 2023 – Mar 2026 sample.
#
# Method: auto.arima() from the {forecast} package automatically selects the
# optimal (p, d, q)(P, D, Q)[12] SARIMA specification based on AIC/BIC.
# The forecast horizon is h = 3 months.
#
# Variables forecasted independently for each country:
#   CPI, IPI, NEER, policy_rate
#
# Global variables (same for both countries) forecasted once:
#   Brent, VIX
#
# The forecasted rows are appended to the observed data and a post_crisis
# dummy (= 1 for March 2026 onwards) is applied to all post-crisis rows.
#
# NOTE: Brent is forecast to remain elevated post-crisis based on the ARIMA
# model fitted on the observed upward spike in March 2026.
# -----------------------------------------------------------------------------

# ── Example: ARIMA forecast for Canada CPI ──────────────────────────────────

# canada_obs <- data_processed %>%
#   filter(country == "Canada", date <= as.Date("2026-03-01"))
#
# cpi_ts_canada <- ts(canada_obs$cpi,
#                     start     = c(2023, 1),
#                     frequency = 12)
#
# arima_cpi_canada <- auto.arima(cpi_ts_canada,
#                                seasonal   = TRUE,
#                                stepwise   = FALSE,
#                                approximation = FALSE)
# summary(arima_cpi_canada)
#
# forecast_cpi_canada <- forecast(arima_cpi_canada, h = 3)
# print(forecast_cpi_canada)
#
# # Extract point forecasts for Apr, May, Jun 2026
# forecasted_cpi_canada <- as.numeric(forecast_cpi_canada$mean)

# ── The same procedure was applied for: ─────────────────────────────────────
#   IPI (Canada & USA), NEER (Canada & USA),
#   policy_rate (Canada & USA), Brent (global), VIX (global)

# ── Post-crisis dummy ────────────────────────────────────────────────────────
# The crisis onset is set to February 27, 2026 (Gulf crisis event date).
# The dummy equals 1 for March 2026 onwards (first full crisis-affected month).

# data_final <- data_processed %>%
#   mutate(
#     post_crisis = ifelse(date >= as.Date("2026-03-01"), 1, 0)
#   )


# -----------------------------------------------------------------------------
# 1E. ASYMMETRIC SHOCK DECOMPOSITION (already in CSV)
# -----------------------------------------------------------------------------
# oil_shock_pos and oil_shock_neg are pre-computed in the CSV.
# Construction:
#   oil_shock_pos_t = oil_pct_change_t  if oil_pct_change_t > 0, else 0
#   oil_shock_neg_t = oil_pct_change_t  if oil_pct_change_t < 0, else 0
# These allow Model Part 9 to test whether price increases and decreases
# have asymmetric effects on CPI and IPI.


# -----------------------------------------------------------------------------
# 1F. FINAL PANEL ASSEMBLY
# -----------------------------------------------------------------------------
# Country metadata appended:
#   oil_balance   = "exporter"          (both Canada and USA are net exporters)
#   country_group = "advanced_exporter" (homogeneous group classification)
#
# Final CSV columns (17 total):
#   country, date, cpi, ipi, neer, policy_rate, brent, vix,
#   ln_cpi, ln_ipi, ln_neer, oil_pct_change, oil_shock_pos, oil_shock_neg,
#   post_crisis, oil_balance, country_group
#
# Dimensions: 84 observations × 17 variables
#   (2 countries × 42 months = balanced panel, no missing country-months)
# -----------------------------------------------------------------------------

cat("NOTE: PART 1 documents data construction only.\n")
cat("The pre-built CSV is loaded in PART 2 below.\n\n")


# =============================================================================
# PART 2 — LOAD THE PRE-BUILT CSV AND INSPECT
# =============================================================================
# The CSV uses comma separators and dates in YYYY-MM-DD format.
# All computed variables (ln_cpi, ln_ipi, ln_neer, oil_pct_change,
# oil_shock_pos, oil_shock_neg, post_crisis) are already present.

data_raw <- read.csv("data/panel_dataset.csv",
                     stringsAsFactors = FALSE)

# Parse date column (format: YYYY-MM-DD as stored in the CSV)
data_raw$date <- as.Date(data_raw$date)

# Basic inspection
str(data_raw)
head(data_raw, 6)

cat("Date range      :", format(min(data_raw$date), "%B %Y"),
    "to", format(max(data_raw$date), "%B %Y"), "\n")
cat("Total obs       :", nrow(data_raw), "\n")
cat("Countries       :", paste(unique(data_raw$country), collapse = " & "), "\n")
cat("Country group   :", unique(data_raw$country_group), "\n")
cat("Oil balance     :", unique(data_raw$oil_balance), "\n")
cat("Post-crisis obs :", sum(data_raw$post_crisis),
    "(Mar–Jun 2026, including ARIMA forecasts)\n")
cat("NAs in oil_pct_change:", sum(is.na(data_raw$oil_pct_change)), "\n\n")

# Define key dates used across all plots
crisis_date <- as.Date("2026-02-27")   # Gulf Crisis onset
arima_start <- as.Date("2026-04-01")   # First ARIMA-forecasted month


# =============================================================================
# PART 3 — EXPLORATORY DATA ANALYSIS & DESCRIPTIVE STATISTICS  [Member 2]
# =============================================================================

# -----------------------------------------------------------------------------
# 3A. DESCRIPTIVE STATISTICS TABLE
# -----------------------------------------------------------------------------
desc_vars <- c("cpi", "ln_cpi", "ipi", "ln_ipi",
               "brent", "oil_pct_change",
               "neer", "ln_neer",
               "policy_rate", "vix")

cat("\n===== DESCRIPTIVE STATISTICS — POOLED (Canada + USA) =====\n")

# Pooled (all 84 observations)
stargazer(data_raw[, desc_vars],
          type         = "text",
          title        = "Table 0. Descriptive Statistics — Canada & USA, Jan 2023–Jun 2026",
          digits       = 4,
          summary.stat = c("n", "mean", "sd", "min", "max"),
          covariate.labels = c(
            "CPI (Index, 2015=100)",    "ln(CPI)",
            "IPI (Index, 2015=100)",    "ln(IPI)",
            "Brent (USD/barrel)",       "Oil Price Change (%)",
            "NEER (Index, 2015=100)",   "ln(NEER)",
            "Policy Rate (%)",          "VIX Index"
          ),
          out = "outputs/Table0_Descriptive_Stats.txt")

cat("Descriptive stats exported → Table0_Descriptive_Stats.txt\n")

# By-country descriptive statistics (printed to console)
cat("\n--- Canada ---\n")
data_raw %>% filter(country == "Canada") %>% select(all_of(desc_vars)) %>%
  summarise(across(everything(),
                   list(N    = ~sum(!is.na(.)),
                        Mean = ~round(mean(., na.rm=TRUE), 4),
                        SD   = ~round(sd(., na.rm=TRUE), 4),
                        Min  = ~round(min(., na.rm=TRUE), 4),
                        Max  = ~round(max(., na.rm=TRUE), 4)))) %>%
  print()

cat("\n--- USA ---\n")
data_raw %>% filter(country == "USA") %>% select(all_of(desc_vars)) %>%
  summarise(across(everything(),
                   list(N    = ~sum(!is.na(.)),
                        Mean = ~round(mean(., na.rm=TRUE), 4),
                        SD   = ~round(sd(., na.rm=TRUE), 4),
                        Min  = ~round(min(., na.rm=TRUE), 4),
                        Max  = ~round(max(., na.rm=TRUE), 4)))) %>%
  print()


# -----------------------------------------------------------------------------
# 3B. PLOT 1 — BRENT CRUDE OIL PRICE OVER TIME
# -----------------------------------------------------------------------------
# Brent is a global variable; we extract one country's rows to avoid doubling.

brent_data <- data_raw %>%
  filter(country == "Canada") %>%
  select(date, brent)

p1 <- ggplot(brent_data, aes(x = date, y = brent)) +
  # ARIMA-forecasted shaded region (April–June 2026)
  annotate("rect",
           xmin = arima_start, xmax = max(brent_data$date),
           ymin = -Inf, ymax = Inf,
           fill = "grey70", alpha = 0.15) +
  annotate("text",
           x = arima_start + 8, y = 65,
           label = "ARIMA\nforecast", color = "grey40",
           size = 3, fontface = "italic", hjust = 0) +
  geom_line(color = "#e67e22", linewidth = 1.2) +
  geom_area(fill = "#e67e22", alpha = 0.10) +
  geom_vline(xintercept = crisis_date,
             linetype = "dashed", color = "#c0392b", linewidth = 0.9) +
  annotate("text",
           x = crisis_date + 10, y = 112,
           label = "Crisis\n(Feb 27, 2026)",
           color = "#c0392b", size = 3.2, fontface = "bold", hjust = 0) +
  labs(title    = "Brent Crude Oil Price: January 2023 – June 2026",
       subtitle = "Dashed line = 2026 Gulf Crisis onset (February 27, 2026)",
       x        = "Date",
       y        = "USD per Barrel",
       caption  = "Sources: U.S. EIA, FRED. Apr–Jun 2026 values: ARIMA forecast.") +
  scale_x_date(date_breaks = "6 months", date_labels = "%b\n%Y") +
  scale_y_continuous(limits = c(55, 120), breaks = seq(55, 120, by = 10)) +
  theme_minimal(base_size = 11) +
  theme(plot.title       = element_text(face = "bold", size = 13),
        plot.subtitle    = element_text(size = 9.5, color = "grey30"),
        plot.caption     = element_text(size = 8, color = "grey40"),
        panel.grid.minor = element_blank())

print(p1)
ggsave("outputs/Plot1_Brent_Oil_Price.png", plot = p1, width = 10, height = 4.8, dpi = 180)
cat("Saved: Plot1_Brent_Oil_Price.png\n")


# -----------------------------------------------------------------------------
# 3C. PLOT 2 — ln(CPI) FOR CANADA & USA OVER TIME
# -----------------------------------------------------------------------------

p2 <- ggplot(data_raw, aes(x = date, y = ln_cpi, color = country, shape = country)) +
  annotate("rect",
           xmin = arima_start, xmax = max(data_raw$date),
           ymin = -Inf, ymax = Inf,
           fill = "grey70", alpha = 0.15) +
  annotate("text",
           x = arima_start + 8, y = min(data_raw$ln_cpi) + 0.002,
           label = "ARIMA\nforecast", color = "grey40",
           size = 3, fontface = "italic", hjust = 0) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 1.8) +
  geom_vline(xintercept = crisis_date,
             linetype = "dashed", color = "#c0392b", linewidth = 0.9) +
  annotate("text",
           x = crisis_date + 10, y = max(data_raw$ln_cpi) - 0.001,
           label = "Crisis\n(Feb 27, 2026)",
           color = "#c0392b", size = 3.2, fontface = "bold", hjust = 0) +
  scale_color_manual(values = c("Canada" = "#2c7bb6", "USA" = "#d7191c"),
                     name = "Country") +
  scale_shape_manual(values = c("Canada" = 16, "USA" = 15),
                     name = "Country") +
  labs(title    = "Log Consumer Price Index — ln(CPI): Canada vs. USA",
       subtitle = "Monthly data, January 2023 – June 2026",
       x        = "Date",
       y        = "ln(CPI)",
       caption  = "Sources: IMF IFS, OECD. Apr–Jun 2026 values: ARIMA forecast.") +
  scale_x_date(date_breaks = "6 months", date_labels = "%b\n%Y") +
  theme_minimal(base_size = 11) +
  theme(plot.title       = element_text(face = "bold", size = 13),
        plot.subtitle    = element_text(size = 9.5, color = "grey30"),
        plot.caption     = element_text(size = 8, color = "grey40"),
        legend.position  = "top",
        panel.grid.minor = element_blank())

print(p2)
ggsave("outputs/Plot2_ln_CPI.png", plot = p2, width = 10, height = 4.8, dpi = 180)
cat("Saved: Plot2_ln_CPI.png\n")


# -----------------------------------------------------------------------------
# 3D. PLOT 3 — ln(IPI) FOR CANADA & USA OVER TIME
# -----------------------------------------------------------------------------

p3 <- ggplot(data_raw, aes(x = date, y = ln_ipi, color = country, shape = country)) +
  annotate("rect",
           xmin = arima_start, xmax = max(data_raw$date),
           ymin = -Inf, ymax = Inf,
           fill = "grey70", alpha = 0.15) +
  annotate("text",
           x = arima_start + 8, y = min(data_raw$ln_ipi) + 0.001,
           label = "ARIMA\nforecast", color = "grey40",
           size = 3, fontface = "italic", hjust = 0) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 1.8) +
  geom_vline(xintercept = crisis_date,
             linetype = "dashed", color = "#c0392b", linewidth = 0.9) +
  annotate("text",
           x = crisis_date + 10, y = min(data_raw$ln_ipi) + 0.003,
           label = "Crisis\n(Feb 27, 2026)",
           color = "#c0392b", size = 3.2, fontface = "bold", hjust = 0) +
  scale_color_manual(values = c("Canada" = "#2c7bb6", "USA" = "#d7191c"),
                     name = "Country") +
  scale_shape_manual(values = c("Canada" = 16, "USA" = 15),
                     name = "Country") +
  labs(title    = "Log Industrial Production Index — ln(IPI): Canada vs. USA",
       subtitle = "Monthly data, January 2023 – June 2026",
       x        = "Date",
       y        = "ln(IPI)",
       caption  = "Sources: IMF IFS, OECD, FRED. Apr–Jun 2026 values: ARIMA forecast.") +
  scale_x_date(date_breaks = "6 months", date_labels = "%b\n%Y") +
  theme_minimal(base_size = 11) +
  theme(plot.title       = element_text(face = "bold", size = 13),
        plot.subtitle    = element_text(size = 9.5, color = "grey30"),
        plot.caption     = element_text(size = 8, color = "grey40"),
        legend.position  = "top",
        panel.grid.minor = element_blank())

print(p3)
ggsave("outputs/Plot3_ln_IPI.png", plot = p3, width = 10, height = 4.8, dpi = 180)
cat("Saved: Plot3_ln_IPI.png\n")


# =============================================================================
# PART 4 — DECLARE THE PANEL DATA FRAME
# =============================================================================
# pdata.frame() declares the panel structure:
#   individual index = country (cross-sectional unit)
#   time index       = date    (time unit)
#
# The first observation per country (January 2023) has oil_pct_change = 0
# (imputed; no prior month). Member 2 noted these may be filtered; here we
# retain them (oil_pct_change = 0) to maintain a fully balanced panel:
#   n = 2 countries,  T = 42 months,  N = 84 total observations.

pdata <- pdata.frame(data_raw, index = c("country", "date"))

cat("\n===== PANEL DIMENSIONS =====\n")
pdim(pdata)
# Expected output: Balanced Panel — n = 2, T = 42, N = 84


# =============================================================================
# PART 5 — CPI INFLATION MODELS  [Member 1]
# =============================================================================
# Dependent variable: ln_cpi
# Regressors: oil_pct_change, ln_neer, policy_rate, vix, post_crisis

# -----------------------------------------------------------------------------
# 5A. MODEL 1 — POOLED OLS (Baseline)
# -----------------------------------------------------------------------------
# Treats all observations as one pool; ignores country-specific heterogeneity.
# Serves as a benchmark only — formally rejected by the Breusch-Pagan test
# in Part 6A.
#
# Equation:
#   ln_cpi_it = B0 + B1*oil_pct_change_t + B2*ln_neer_it + B3*policy_rate_it
#               + B4*vix_t + B5*post_crisis_t + e_it

pols_model <- plm(ln_cpi ~ oil_pct_change + ln_neer + policy_rate + vix + post_crisis,
                  data  = pdata,
                  model = "pooling")

cat("\n===== MODEL 1 — POOLED OLS — CPI =====\n")
summary(pols_model)
# R² = 0.253. Oil price change insignificant (p = 0.901). Only ln_neer is
# significant (B = -0.448, p = 0.016). Post-crisis marginally significant
# (B = 0.040, p = 0.052). Low explanatory power confirms country-specific
# heterogeneity is being ignored. This model is a baseline only.


# -----------------------------------------------------------------------------
# 5B. MODEL 2 — FIXED EFFECTS (Within Estimator)
# -----------------------------------------------------------------------------
# Each country receives its own time-invariant intercept (alpha_i), absorbing
# all unobserved structural differences between Canada and the USA
# (energy market integration, industrial mix, monetary policy credibility, etc.)
#
# Equation:
#   ln_cpi_it = alpha_i + B1*oil_pct_change_t + B2*ln_neer_it
#               + B3*policy_rate_it + B4*vix_t + B5*post_crisis_t + e_it
#
# Argument model = "within" is the plm syntax for Fixed Effects.

fe_model <- plm(ln_cpi ~ oil_pct_change + ln_neer + policy_rate + vix + post_crisis,
                data  = pdata,
                model = "within")

cat("\n===== MODEL 2 — FIXED EFFECTS — CPI =====\n")
summary(fe_model)
# R² = 0.647 — major improvement over POLS.
# Policy rate: B = -0.023, p < 0.001 (monetary tightening reduces inflation).
# Post-crisis dummy: B = 0.041, p < 0.001 (CPI ~4.1% higher during crisis).
# Oil price change: B = -0.0003, p = 0.187 (not significant at standard levels
# before robust SE; consistent with net-exporter status of Canada & USA).


# -----------------------------------------------------------------------------
# 5C. MODEL 3 — RANDOM EFFECTS (Attempted — Expected to Fail)
# -----------------------------------------------------------------------------
# The professor's framework requires attempting the Random Effects estimator.
# RE assumes country-specific effects are RANDOM draws uncorrelated with
# regressors, and uses a Generalised Least Squares (GLS) transformation.
#
# WHY RE FAILS WITH N = 2 COUNTRIES:
# ─────────────────────────────────────────────────────────────────────────────
# The RE estimator must estimate:
#   (a) the regression coefficients (k parameters), AND
#   (b) the variance of the random effect sigma²_u separately from the
#       idiosyncratic error variance sigma²_e.
#
# With only N = 2 cross-sectional units (Canada and USA), the between-group
# variation — the only information RE uses to identify sigma²_u — comes from
# a single degree of freedom (N − 1 = 1). This is insufficient to separately
# identify the random effects variance from the regression parameters.
# The GLS estimator collapses: the estimated theta (quasi-demeaning parameter)
# becomes undefined or degenerate, causing plm() to throw an error.
#
# This is NOT a data quality problem — it is a fundamental mathematical
# constraint of the RE estimator that requires N ≥ 5 (practically N ≥ 10)
# cross-sectional units to produce reliable estimates.
#
# CONSEQUENCE:
#   • RE is eliminated by construction.
#   • The Hausman test (which requires both FE and RE estimates) is also
#     infeasible (see Part 6B).
#   • Fixed Effects is confirmed as the ONLY valid panel estimator for this
#     2-country panel. This is further corroborated by the Breusch-Pagan
#     test in Part 6A (p < 2.2e-16), which rejects POLS decisively.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n===== MODEL 3 — RANDOM EFFECTS (Attempted) =====\n")
cat("Attempting RE estimation with tryCatch() to document the failure...\n\n")

re_model <- tryCatch({

  # Attempt RE estimation — plm argument model = "random"
  m <- plm(ln_cpi ~ oil_pct_change + ln_neer + policy_rate + vix + post_crisis,
            data  = pdata,
            model = "random")
  cat("RE model estimated successfully (unexpected with N=2).\n")
  summary(m)
  m   # return the object if it somehow succeeds

}, error = function(e) {

  # ── Caught error: print full academic explanation ────────────────────────
  cat("--------------------------------------------------------------\n")
  cat("RESULT: Random Effects estimation FAILED — as expected.\n")
  cat("ERROR MESSAGE FROM plm():\n")
  cat(" ", conditionMessage(e), "\n\n")
  cat("ACADEMIC EXPLANATION:\n")
  cat("  The RE (GLS) estimator requires estimating the random-effects\n")
  cat("  variance component sigma²_u from the BETWEEN-group variation.\n")
  cat("  With only N = 2 entities (Canada, USA), the between-dimension\n")
  cat("  has N - 1 = 1 degree of freedom — insufficient to separately\n")
  cat("  identify sigma²_u from the k regression parameters.\n")
  cat("  The quasi-demeaning parameter theta = 1 - sqrt(sigma²_e /\n")
  cat("  (sigma²_e + T*sigma²_u)) becomes degenerate, causing the\n")
  cat("  GLS transformation to fail.\n\n")
  cat("CONCLUSION:\n")
  cat("  RE is eliminated by mathematical necessity, not by choice.\n")
  cat("  Fixed Effects (Part 5B) is the only valid panel estimator.\n")
  cat("  This is consistent with the Breusch-Pagan test (Part 6A)\n")
  cat("  which rejects POLS with chi²(1) = 278.58, p < 2.2e-16,\n")
  cat("  confirming that individual effects must be controlled for.\n")
  cat("--------------------------------------------------------------\n")
  NULL   # re_model set to NULL — used in Part 6B Hausman check

})

# Confirm RE object status for downstream code
if (is.null(re_model)) {
  cat("\nre_model = NULL  (RE estimation failed, as documented above)\n")
}


# =============================================================================
# PART 6 — DIAGNOSTIC TESTS  [Member 1]
# =============================================================================

# -----------------------------------------------------------------------------
# 6A. BREUSCH-PAGAN LM TEST — POLS vs PANEL MODELS
# -----------------------------------------------------------------------------
# H0: No individual country effects; Pooled OLS is sufficient.
# H1: Significant individual effects exist; panel estimation is needed.
# Decision rule: p < 0.05 → reject H0 → POLS is invalid.

cat("\n===== DIAGNOSTIC TEST: BREUSCH-PAGAN LM TEST =====\n")
bp_test <- plmtest(pols_model, effect = "individual", type = "bp")
print(bp_test)
# Expected result: chi²(1) = 278.58, p < 2.2e-16 → H0 rejected decisively.
# Conclusion: Strong individual country effects exist.
# POLS is invalid; panel estimators (FE) are required.


# -----------------------------------------------------------------------------
# 6B. HAUSMAN TEST — phtest(fe_model, re_model) — Attempted
# -----------------------------------------------------------------------------
# The Hausman (1978) specification test compares the FE and RE coefficient
# vectors to decide which estimator is appropriate.
# H0: RE is consistent and efficient (country effects uncorrelated with X).
# H1: RE is inconsistent; only FE is consistent.
# Decision rule: p < 0.05 → reject H0 → prefer Fixed Effects.
#
# WHY THE HAUSMAN TEST CANNOT BE CONDUCTED HERE:
# ─────────────────────────────────────────────────────────────────────────────
# The Hausman statistic is:
#   H = (b_FE − b_RE)' [Var(b_FE) − Var(b_RE)]^{-1} (b_FE − b_RE)
# This requires a valid RE estimate (b_RE). Since RE failed in Part 5C
# because N = 2 countries cannot identify sigma²_u (only 1 between-group
# degree of freedom), there is no b_RE to compare against.
# This is a cascading failure: RE fails → Hausman input is missing → test fails.
#
# ALTERNATIVE JUSTIFICATION FOR FE (without Hausman):
#   (1) Theory: Canada and USA have structural differences (energy markets,
#       central bank mandates, industrial mix) almost certainly correlated
#       with regressors — violating RE's orthogonality assumption.
#   (2) Breusch-Pagan (Part 6A): chi²(1)=278.58, p < 2.2e-16 → POLS rejected.
#       With RE infeasible, FE is the only valid estimator by elimination.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n===== DIAGNOSTIC TEST: HAUSMAN TEST (Attempted) =====\n")
cat("Attempting phtest(fe_model, re_model) with tryCatch()...\n\n")

hausman_test <- tryCatch({

  if (is.null(re_model)) {
    stop(paste(
      "re_model is NULL — RE estimation failed in Part 5C (N=2 countries).",
      "Hausman test requires a valid RE estimate as input.",
      "Cannot compute H = (b_FE - b_RE)' [V_FE - V_RE]^{-1} (b_FE - b_RE)."
    ))
  }
  ht <- phtest(fe_model, re_model)
  cat("Hausman test completed (unexpected with N=2).\n")
  print(ht)
  ht

}, error = function(e) {

  cat("--------------------------------------------------------------\n")
  cat("RESULT: Hausman test FAILED — as expected with N=2 countries.\n")
  cat("ERROR MESSAGE:\n  ", conditionMessage(e), "\n\n")
  cat("ACADEMIC EXPLANATION:\n")
  cat("  The Hausman test statistic requires b_RE (RE coefficient vector)\n")
  cat("  as input. Since RE could not be estimated (sigma²_u unidentifiable\n")
  cat("  with N-1=1 between-group degree of freedom), b_RE does not exist.\n")
  cat("  The test cannot be computed. This is a cascading consequence of\n")
  cat("  the fundamental N=2 constraint, not a data or coding error.\n\n")
  cat("CONCLUSION: FE selected by (1) theoretical necessity and\n")
  cat("  (2) Breusch-Pagan elimination of POLS (p < 2.2e-16).\n")
  cat("--------------------------------------------------------------\n")
  NULL

})


# -----------------------------------------------------------------------------
# 6C. AUTOCORRELATION TEST — BREUSCH-GODFREY / WOOLDRIDGE
# -----------------------------------------------------------------------------
# H0: No serial correlation in panel residuals.
# H1: Serial correlation present.
# Decision rule: p < 0.05 → reject H0 → robust SEs required.

cat("\n===== DIAGNOSTIC TEST: AUTOCORRELATION (Wooldridge/pbgtest) =====\n")
autocorr_test <- pbgtest(fe_model)
print(autocorr_test)
# Expected result: chi²(42) = 75.301, p = 0.0012 → H0 rejected.
# Serial correlation confirmed — expected for monthly macro data (inflation
# is naturally persistent). Standard SEs are underestimated.
# → Justifies use of robust standard errors in Part 7.


# =============================================================================
# PART 7 — ROBUST STANDARD ERRORS FOR CPI MODELS  [Member 1]
# =============================================================================
# Arellano (1987) HC3 robust SEs correct for heteroskedasticity AND
# serial autocorrelation. Applied via vcovHC() + coeftest().
# Coefficients are unchanged; only SEs and p-values are updated.
#
# Cluster-robust SEs by country are NOT used: with N = 2 clusters, the
# asymptotic justification requires N ≥ ~10, so HC3 robust SE is the
# appropriate and standard choice for small-N panels.

# -----------------------------------------------------------------------------
# 7A. ARELLANO HC3 ROBUST STANDARD ERRORS
# -----------------------------------------------------------------------------
robust_vcov_pols <- vcovHC(pols_model, method = "arellano", type = "HC3")
robust_vcov_fe   <- vcovHC(fe_model,   method = "arellano", type = "HC3")

cat("\n--- POLS: Robust Standard Errors (Arellano HC3) ---\n")
coeftest(pols_model, vcov = robust_vcov_pols)
# With robust SEs: oil_pct_change becomes spuriously significant with a
# negative sign — theoretically incorrect for an inflation model, further
# demonstrating POLS unreliability. Post-crisis dummy: B = 0.040, p < 0.001.

cat("\n--- FIXED EFFECTS: Robust Standard Errors (Arellano HC3) ---\n")
coeftest(fe_model, vcov = robust_vcov_fe)
# With robust SEs, all five regressors become significant:
#   oil_pct_change: B = -0.0003, p < 0.001 (small negative — net-exporter effect)
#   ln_neer       : B =  0.254,  p = 0.016
#   policy_rate   : B = -0.023,  p < 0.001 (monetary tightening reduces CPI)
#   vix           : B = -0.001,  p < 0.001 (uncertainty suppresses prices)
#   post_crisis   : B =  0.041,  p < 0.001 ← central finding of the study


# -----------------------------------------------------------------------------
# 7B. STARGAZER TABLE — POLS AND FIXED EFFECTS SIDE-BY-SIDE  (Table 1)
# -----------------------------------------------------------------------------
se_pols <- sqrt(diag(robust_vcov_pols))
se_fe   <- sqrt(diag(robust_vcov_fe))

# Console output
stargazer(pols_model, fe_model,
          title          = "Table 1: CPI Inflation Models — Pooled OLS and Fixed Effects",
          column.labels  = c("Pooled OLS", "Fixed Effects"),
          dep.var.labels = "ln(CPI)",
          covariate.labels = c("Oil Price Change (%)",
                               "ln(NEER)",
                               "Policy Rate (%)",
                               "VIX Index",
                               "Post-Crisis Dummy (=1 if date ≥ Mar 2026)"),
          se             = list(se_pols, se_fe),
          type           = "text",
          keep.stat      = c("n", "rsq", "adj.rsq"),
          notes          = "Robust SE (Arellano HC3) in parentheses. *, **, *** = 10%, 5%, 1%.",
          notes.append   = FALSE,
          star.cutoffs   = c(0.10, 0.05, 0.01))

# HTML export
stargazer(pols_model, fe_model,
          title          = "Table 1: CPI Inflation Models — Pooled OLS and Fixed Effects",
          column.labels  = c("Pooled OLS", "Fixed Effects"),
          dep.var.labels = "ln(CPI)",
          covariate.labels = c("Oil Price Change (%)",
                               "ln(NEER)",
                               "Policy Rate (%)",
                               "VIX Index",
                               "Post-Crisis Dummy (=1 if date ≥ Mar 2026)"),
          se             = list(se_pols, se_fe),
          type           = "html",
          out            = "outputs/Table1_CPI_Models.html",
          keep.stat      = c("n", "rsq", "adj.rsq"),
          notes          = "Robust SE (Arellano HC3) in parentheses. *, **, *** = 10%, 5%, 1%.",
          notes.append   = FALSE,
          star.cutoffs   = c(0.10, 0.05, 0.01))

cat("Exported: Table1_CPI_Models.html\n")


# =============================================================================
# PART 8 — IPI INDUSTRIAL PRODUCTION MODEL  [Member 2]
# =============================================================================
# Dependent variable: ln_ipi  ("growth equation")
# Same regressors as the CPI model.
# Expected signs from theory:
#   oil_pct_change: negative (oil price increases reduce industrial output)
#   ln_neer       : ambiguous (stronger currency may hurt exporters or lower input costs)
#   policy_rate   : negative (contractionary monetary policy reduces investment)
#   vix           : negative (uncertainty suppresses investment and production)
#   post_crisis   : negative (crisis period expected to depress industrial activity)

# -----------------------------------------------------------------------------
# 8A. FIXED EFFECTS MODEL — ln(IPI)
# -----------------------------------------------------------------------------

fe_ipi <- plm(ln_ipi ~ oil_pct_change + ln_neer + policy_rate + vix + post_crisis,
              data  = pdata,
              model = "within")

cat("\n===== MODEL — FIXED EFFECTS — IPI =====\n")
summary(fe_ipi)


# -----------------------------------------------------------------------------
# 8B. ARELLANO HC3 ROBUST STANDARD ERRORS — IPI
# -----------------------------------------------------------------------------
# Same HC3 procedure as CPI; see Part 7 for rationale.

robust_vcov_ipi <- vcovHC(fe_ipi, method = "arellano", type = "HC3")
robust_se_ipi   <- sqrt(diag(robust_vcov_ipi))

cat("\n--- FIXED EFFECTS IPI: Robust Standard Errors (Arellano HC3) ---\n")
coeftest(fe_ipi, vcov = robust_vcov_ipi)


# -----------------------------------------------------------------------------
# 8C. AUTOCORRELATION TEST — IPI MODEL
# -----------------------------------------------------------------------------
cat("\n===== AUTOCORRELATION TEST: IPI MODEL =====\n")
pbgtest(fe_ipi)
# H0: No serial correlation. p < 0.05 → reject H0 → confirms need for robust SEs.


# -----------------------------------------------------------------------------
# 8D. NEWEY-WEST HAC SE — ROBUSTNESS CHECK
# -----------------------------------------------------------------------------
# Newey-West HAC standard errors simultaneously correct for both
# heteroskedasticity and serial autocorrelation.
# Number of lags: 4 (rule of thumb: floor(T^(1/4)) for T ≈ 42).
# This serves as a robustness check on the primary Arellano HC3 results.

nw_vcov_ipi <- NeweyWest(fe_ipi, lag = 4, prewhite = FALSE)
nw_se_ipi   <- sqrt(diag(nw_vcov_ipi))

cat("\n--- IPI FE MODEL: Newey-West HAC SE (4 lags) — Robustness Check ---\n")
coeftest(fe_ipi, vcov = nw_vcov_ipi)


# -----------------------------------------------------------------------------
# 8E. STARGAZER TABLES — IPI (Table 2)
# -----------------------------------------------------------------------------

# Primary table: HC3 robust SE
stargazer(fe_ipi,
          se       = list(robust_se_ipi),
          title    = "Table 2: Fixed Effects Model — Dependent Variable: ln(IPI)",
          dep.var.labels   = "ln(Industrial Production Index)",
          covariate.labels = c("Oil Price Change (%)",
                               "ln(NEER)",
                               "Policy Rate (%)",
                               "VIX Index",
                               "Post-Crisis Dummy (=1 if date ≥ Mar 2026)"),
          add.lines = list(
            c("Country Fixed Effects",   "Yes"),
            c("Robust SE (Arellano HC3)", "Yes")
          ),
          omit.stat = c("f", "ser"),
          type      = "text",
          digits    = 6,
          notes     = c("* p<0.10, ** p<0.05, *** p<0.01",
                        "HC3 robust SE in parentheses.",
                        "Panel: Canada & USA, Jan 2023–Jun 2026 (N=84).",
                        "Crisis onset: February 27, 2026."),
          out = "outputs/Table2_IPI_FE_Model.txt")

cat("Exported: Table2_IPI_FE_Model.txt\n")

# Robustness comparison table: HC3 vs Newey-West
stargazer(fe_ipi, fe_ipi,
          se       = list(robust_se_ipi, nw_se_ipi),
          title    = "Table 2b: IPI FE Model — HC3 vs Newey-West SE Comparison",
          dep.var.labels   = c("ln(IPI) [HC3]", "ln(IPI) [NW-HAC]"),
          covariate.labels = c("Oil Price Change (%)", "ln(NEER)",
                               "Policy Rate (%)", "VIX Index",
                               "Post-Crisis Dummy"),
          add.lines = list(
            c("Country Fixed Effects", "Yes", "Yes"),
            c("SE Method", "Arellano HC3", "Newey-West HAC (4 lags)")
          ),
          omit.stat = c("f", "ser"),
          type      = "text",
          digits    = 6,
          notes     = c("* p<0.10, ** p<0.05, *** p<0.01",
                        "Monthly panel: Canada & USA, Jan 2023–Jun 2026.",
                        "NW-HAC: 4 lags (floor(T^1/4) rule)."),
          out = "outputs/Table2b_IPI_HC3_vs_NWHAC.txt")

cat("Exported: Table2b_IPI_HC3_vs_NWHAC.txt\n")

# Interpretation guide
cat("\n===== COEFFICIENT INTERPRETATION — IPI Model =====\n")
cat("All variables are in log or percentage form (semi-elasticities).\n")
cat("  oil_pct_change : 1 pp ↑ oil price → [coef×100]% change in IPI\n")
cat("  ln_neer        : 1% ↑ NEER → [coef]% change in IPI (log-log elasticity)\n")
cat("  policy_rate    : 1 pp ↑ policy rate → [coef×100]% change in IPI\n")
cat("  vix            : 1-unit ↑ VIX → [coef×100]% change in IPI\n")
cat("  post_crisis    : Being in crisis period → [coef×100]% change in IPI\n\n")


# =============================================================================
# PART 9 — ASYMMETRY ANALYSIS  [Member 3]
# =============================================================================
# Standard models assume symmetric effects: a 5% oil price increase affects
# CPI and IPI in equal-and-opposite magnitude to a 5% decrease.
# Economic theory (Mork 1989, Hamilton 2003) suggests this is unlikely:
#   — price increases may be inflationary and output-reducing,
#   — price decreases may be absorbed by firms rather than passed on.
#
# We test this by splitting oil_pct_change into:
#   oil_shock_pos = positive changes only (already in the CSV)
#   oil_shock_neg = negative changes only (already in the CSV)
# A Wald test on equality of coefficients formally tests asymmetry.

# Replace NA in shock columns (first obs per country = 0)
data_raw <- data_raw %>%
  group_by(country) %>%
  mutate(
    oil_shock_pos = ifelse(is.na(oil_shock_pos), 0, oil_shock_pos),
    oil_shock_neg = ifelse(is.na(oil_shock_neg), 0, oil_shock_neg)
  ) %>%
  ungroup()

pdata <- pdata.frame(data_raw, index = c("country", "date"))


# -----------------------------------------------------------------------------
# 9A. ASYMMETRIC FE MODEL — ln(CPI)
# -----------------------------------------------------------------------------

fe_asym_cpi <- plm(
  ln_cpi ~ oil_shock_pos + oil_shock_neg + ln_neer + policy_rate + vix + post_crisis,
  data  = pdata,
  model = "within"
)

cat("\n===== ASYMMETRIC FE MODEL — CPI =====\n")
summary(fe_asym_cpi)

cat("\n--- Asymmetric CPI: Robust SE (Arellano HC3) ---\n")
coeftest(fe_asym_cpi, vcov = vcovHC(fe_asym_cpi, method = "arellano", type = "HC3"))

# -----------------------------------------------------------------------------
# WALD TEST — Formal test of asymmetry in CPI model
# -----------------------------------------------------------------------------
# H0: The effect of positive oil shocks = effect of negative oil shocks
#     (i.e., symmetric transmission: coef(oil_shock_pos) = coef(oil_shock_neg))
# H1: The effects are different (asymmetric transmission)
# Decision rule: p < 0.05 → reject H0 → asymmetric pass-through confirmed.
# Reference: Mork (1989); Hamilton (2003) — cited in report Section 10.1.

cat("\n--- WALD TEST: Asymmetry in CPI model (H0: pos = neg) ---\n")
wald_asym_cpi <- linearHypothesis(fe_asym_cpi, "oil_shock_pos = oil_shock_neg")
print(wald_asym_cpi)
# If p > 0.05: fail to reject H0 → no statistically significant asymmetry detected.
# Consistent with report Section 10.1 finding of no detectable asymmetry.


# -----------------------------------------------------------------------------
# 9B. ASYMMETRIC FE MODEL — ln(IPI)
# -----------------------------------------------------------------------------

fe_asym_ipi <- plm(
  ln_ipi ~ oil_shock_pos + oil_shock_neg + ln_neer + policy_rate + vix + post_crisis,
  data  = pdata,
  model = "within"
)

cat("\n===== ASYMMETRIC FE MODEL — IPI =====\n")
summary(fe_asym_ipi)

cat("\n--- Asymmetric IPI: Robust SE (Arellano HC3) ---\n")
coeftest(fe_asym_ipi, vcov = vcovHC(fe_asym_ipi, method = "arellano", type = "HC3"))

# -----------------------------------------------------------------------------
# WALD TEST — Formal test of asymmetry in IPI model
# -----------------------------------------------------------------------------
# H0: coef(oil_shock_pos) = coef(oil_shock_neg) (symmetric transmission)
# H1: Asymmetric transmission of oil shocks to industrial production.

cat("\n--- WALD TEST: Asymmetry in IPI model (H0: pos = neg) ---\n")
wald_asym_ipi <- linearHypothesis(fe_asym_ipi, "oil_shock_pos = oil_shock_neg")
print(wald_asym_ipi)
# Reference: Mork (1989); Hamilton (2003) — cited in report Section 10.1.
# If p > 0.05: no statistically significant asymmetry — consistent with
# report Section 10.1 conclusion of no detectable asymmetry for advanced exporters.


# -----------------------------------------------------------------------------
# 9C. STARGAZER TABLE — ASYMMETRY (Table 3)
# -----------------------------------------------------------------------------

stargazer(
  fe_asym_cpi, fe_asym_ipi,
  title         = "Table 3: Asymmetric Oil Shock Effects — FE Models",
  column.labels = c("ln(CPI) — Inflation", "ln(IPI) — Industrial Production"),
  covariate.labels = c(
    "Oil Shock: Positive (↑ only, %)",
    "Oil Shock: Negative (↓ only, %)",
    "ln(NEER)",
    "Policy Rate (%)",
    "VIX Index",
    "Post-Crisis Dummy"
  ),
  add.lines = list(c("Country Fixed Effects", "Yes", "Yes")),
  keep.stat = c("n", "rsq", "adj.rsq"),
  type      = "text",
  notes     = c("* p<0.10, ** p<0.05, *** p<0.01",
                "Arellano HC3 robust SE in parentheses.",
                "If coef(pos) ≠ coef(neg), asymmetric pass-through is present."),
  out       = "outputs/Table3_Asymmetry_CPI_IPI.txt"
)

cat("Exported: Table3_Asymmetry_CPI_IPI.txt\n")


# =============================================================================
# PART 10 — HETEROGENEITY ANALYSIS  [Member 3 + Professor PDF]
# =============================================================================
# The professor's framework requires splitting the panel into Oil Importers
# vs Oil Exporters and running separate FE models for each subgroup.
# We first attempt this split using the oil_balance column (as the PDF
# prescribes), document why it fails, then proceed to the only feasible
# heterogeneity analysis available: country-level sub-regressions
# (Canada vs USA separately).

# -----------------------------------------------------------------------------
# 10A. HETEROGENEITY: OIL IMPORTERS vs OIL EXPORTERS (Attempted)
# -----------------------------------------------------------------------------
# The professor's specification:
#   importers_data <- pdata %>% filter(oil_balance == "importer")
#   exporters_data <- pdata %>% filter(oil_balance == "exporter")
#   fe_importers   <- plm(ln_cpi ~ ..., data = importers_data, model = "within")
#   fe_exporters   <- plm(ln_cpi ~ ..., data = exporters_data, model = "within")
#
# WHY THIS SPLIT FAILS WITH OUR PANEL:
# ─────────────────────────────────────────────────────────────────────────────
# Both Canada and the USA are classified as net oil EXPORTERS
# (oil_balance = "exporter" for all 84 observations).
# This means:
#   • The "importers" subset is EMPTY (0 observations).
#   • The "exporters" subset contains ALL 84 observations (both countries).
#
# Running FE on the exporters subset replicates the full-panel FE model
# exactly (no new information). Running FE on the importers subset is
# impossible (no data).
#
# ROOT CAUSE: Our research design deliberately selected Canada and USA as
# a homogeneous group of advanced net-exporters to study how oil-exporting
# economies respond to an oil price shock. The importer/exporter split
# requires a heterogeneous sample (e.g., adding oil-importing economies
# like Japan, Germany, India). With only 2 countries from the same group,
# the split is structurally degenerate.
#
# CONSEQUENCE: The heterogeneity dimension in our panel is COUNTRY
# (Canada vs USA), not IMPORTER vs EXPORTER. The country-level regressions
# in Parts 10B–10C are the appropriate substitute.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n===== HETEROGENEITY: OIL IMPORTERS vs EXPORTERS (Attempted) =====\n")

# Check the distribution of oil_balance in our data
cat("Distribution of oil_balance in the panel:\n")
print(table(data_raw$oil_balance))
cat("\n")

# Attempt the importer/exporter split with tryCatch
cat("Attempting FE model on importers subset...\n")
fe_importers <- tryCatch({

  importers_data <- pdata[pdata$oil_balance == "importer", ]
  cat("Importer observations found:", nrow(importers_data), "\n")

  if (nrow(importers_data) == 0) {
    stop(paste(
      "The importers subset is EMPTY (0 observations).",
      "Both Canada and USA have oil_balance = 'exporter'.",
      "A Fixed Effects model requires at least 2 entities and",
      "sufficient observations. No importer data exists in this panel."
    ))
  }
  imp_pdata <- pdata.frame(importers_data, index = c("country", "date"))
  plm(ln_cpi ~ oil_pct_change + ln_neer + policy_rate + vix + post_crisis,
      data = imp_pdata, model = "within")

}, error = function(e) {
  cat("--------------------------------------------------------------\n")
  cat("RESULT: Importers FE model FAILED — as expected.\n")
  cat("ERROR:\n  ", conditionMessage(e), "\n\n")
  cat("EXPLANATION:\n")
  cat("  Both Canada and USA are net oil exporters (oil_balance = 'exporter').\n")
  cat("  The importer subset contains 0 rows — FE estimation is impossible.\n")
  cat("  This is a structural feature of our research design, not an error.\n")
  cat("  Our panel was intentionally built from a homogeneous exporter group.\n")
  cat("  Adding oil-importing economies would require expanding the sample.\n")
  cat("--------------------------------------------------------------\n")
  NULL
})

cat("\nAttempting FE model on exporters subset...\n")
fe_exporters <- tryCatch({

  exporters_data <- pdata[pdata$oil_balance == "exporter", ]
  cat("Exporter observations found:", nrow(exporters_data), "\n")

  exp_pdata <- pdata.frame(as.data.frame(exporters_data),
                           index = c("country", "date"))
  m <- plm(ln_cpi ~ oil_pct_change + ln_neer + policy_rate + vix + post_crisis,
           data = exp_pdata, model = "within")
  cat("Exporters FE model estimated (all 84 obs = full panel replication).\n")
  summary(m)
  m

}, error = function(e) {
  cat("Exporters FE error:", conditionMessage(e), "\n")
  NULL
})

cat("\n--- IMPORTER vs EXPORTER SPLIT — SUMMARY ---\n")
cat("Importers model : NULL (empty subset — 0 importer countries in panel)\n")
cat("Exporters model : Identical to full-panel FE (all N=84 obs are exporters)\n")
cat("CONCLUSION      : The importer/exporter split is degenerate for this panel.\n")
cat("                  Country-level heterogeneity (Canada vs USA) is analysed\n")
cat("                  in Parts 10B–10C below, which is the appropriate\n")
cat("                  substitute given our homogeneous 2-country exporter panel.\n\n")

# -----------------------------------------------------------------------------
# 10B. COUNTRY-LEVEL HETEROGENEITY: CANADA — CPI AND IPI
# -----------------------------------------------------------------------------
# Since the importer/exporter split fails (Part 10A), the feasible heterogeneity
# analysis compares Canada vs USA directly via separate time-series regressions.
# With only 1 entity per sub-panel, model = "pooling" is used (equivalent to OLS;
# no cross-sectional fixed effects are needed or estimable with N=1).

canada_data <- pdata.frame(
  data_raw %>% filter(country == "Canada"),
  index = c("country", "date")
)

# Canada — CPI
fe_canada_cpi <- plm(
  ln_cpi ~ oil_pct_change + ln_neer + policy_rate + vix + post_crisis,
  data  = canada_data,
  model = "pooling"   # N=1 entity → pooling = OLS (cross-sectional FE not applicable)
)

cat("\n===== COUNTRY-LEVEL: CANADA — CPI =====\n")
summary(fe_canada_cpi)

# Canada — IPI
fe_canada_ipi <- plm(
  ln_ipi ~ oil_pct_change + ln_neer + policy_rate + vix + post_crisis,
  data  = canada_data,
  model = "pooling"
)

cat("\n===== COUNTRY-LEVEL: CANADA — IPI =====\n")
summary(fe_canada_ipi)


# -----------------------------------------------------------------------------
# 10C. COUNTRY-LEVEL HETEROGENEITY: USA — CPI AND IPI
# -----------------------------------------------------------------------------

usa_data <- pdata.frame(
  data_raw %>% filter(country == "USA"),
  index = c("country", "date")
)

# USA — CPI
fe_usa_cpi <- plm(
  ln_cpi ~ oil_pct_change + ln_neer + policy_rate + vix + post_crisis,
  data  = usa_data,
  model = "pooling"
)

cat("\n===== COUNTRY-LEVEL: USA — CPI =====\n")
summary(fe_usa_cpi)

# USA — IPI
fe_usa_ipi <- plm(
  ln_ipi ~ oil_pct_change + ln_neer + policy_rate + vix + post_crisis,
  data  = usa_data,
  model = "pooling"
)

cat("\n===== COUNTRY-LEVEL: USA — IPI =====\n")
summary(fe_usa_ipi)


# -----------------------------------------------------------------------------
# 10D. STARGAZER TABLES — CANADA vs USA + IMPORTER/EXPORTER NOTE (Tables 4 & 5)
# -----------------------------------------------------------------------------
# NOTE: The professor's guide calls for a stargazer table comparing importers
# vs exporters FE models. Since that split is degenerate (Part 10A), we
# substitute the equivalent comparison: Canada vs USA within the exporter group.
# This is academically valid and more informative for our research question.

# Table 4: Country comparison — CPI
stargazer(
  fe_canada_cpi, fe_usa_cpi,
  title         = "Table 4: Country-Level Results — Impact on Inflation ln(CPI)",
  column.labels = c("Canada", "USA"),
  dep.var.labels = "ln(CPI)",
  covariate.labels = c("Oil Price Change (%)", "ln(NEER)",
                       "Policy Rate (%)", "VIX Index", "Post-Crisis Dummy"),
  keep.stat = c("n", "rsq", "adj.rsq"),
  type      = "text",
  notes     = c("* p<0.10, ** p<0.05, *** p<0.01",
                "Country-level OLS (single-entity time-series for each country)."),
  out       = "outputs/Table4_Country_CPI_Canada_vs_USA.txt"
)

cat("Exported: Table4_Country_CPI_Canada_vs_USA.txt\n")

# Table 5: Country comparison — IPI
stargazer(
  fe_canada_ipi, fe_usa_ipi,
  title         = "Table 5: Country-Level Results — Impact on Industrial Production ln(IPI)",
  column.labels = c("Canada", "USA"),
  dep.var.labels = "ln(IPI)",
  covariate.labels = c("Oil Price Change (%)", "ln(NEER)",
                       "Policy Rate (%)", "VIX Index", "Post-Crisis Dummy"),
  keep.stat = c("n", "rsq", "adj.rsq"),
  type      = "text",
  notes     = c("* p<0.10, ** p<0.05, *** p<0.01",
                "Country-level OLS (single-entity time-series for each country)."),
  out       = "outputs/Table5_Country_IPI_Canada_vs_USA.txt"
)

cat("Exported: Table5_Country_IPI_Canada_vs_USA.txt\n")


# =============================================================================
# PART 11 — SCATTER PLOTS  [Member 3]
# =============================================================================

# -----------------------------------------------------------------------------
# 11A. SCATTER: OIL PRICE CHANGE vs ln(CPI)
# -----------------------------------------------------------------------------

p4 <- ggplot(data_raw, aes(x = oil_pct_change, y = ln_cpi, color = country)) +
  geom_point(size = 2.2, alpha = 0.75) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 0.8) +
  scale_color_manual(values = c("Canada" = "#C0392B", "USA" = "#2980B9")) +
  geom_vline(xintercept = 0, linetype = "dashed",
             color = "grey50", linewidth = 0.4) +
  labs(title    = "Oil Price Shock vs Log CPI — Canada & USA",
       subtitle = "January 2023 – June 2026 | Each point = one country-month",
       x        = "Monthly Oil Price Change (%)",
       y        = "ln(CPI)",
       color    = "Country") +
  theme_minimal(base_size = 12) +
  theme(plot.title       = element_text(face = "bold", size = 13),
        plot.subtitle    = element_text(color = "grey40", size = 10),
        legend.position  = "bottom")

ggsave("outputs/Plot4_Scatter_Oil_vs_lnCPI.png", plot = p4,
       width = 8, height = 5, dpi = 300)
print(p4)
cat("Saved: Plot4_Scatter_Oil_vs_lnCPI.png\n")


# -----------------------------------------------------------------------------
# 11B. SCATTER: OIL PRICE CHANGE vs ln(IPI)
# -----------------------------------------------------------------------------

p5 <- ggplot(data_raw, aes(x = oil_pct_change, y = ln_ipi, color = country)) +
  geom_point(size = 2.2, alpha = 0.75) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 0.8) +
  scale_color_manual(values = c("Canada" = "#C0392B", "USA" = "#2980B9")) +
  geom_vline(xintercept = 0, linetype = "dashed",
             color = "grey50", linewidth = 0.4) +
  labs(title    = "Oil Price Shock vs Log Industrial Production — Canada & USA",
       subtitle = "January 2023 – June 2026 | Each point = one country-month",
       x        = "Monthly Oil Price Change (%)",
       y        = "ln(IPI)",
       color    = "Country") +
  theme_minimal(base_size = 12) +
  theme(plot.title       = element_text(face = "bold", size = 13),
        plot.subtitle    = element_text(color = "grey40", size = 10),
        legend.position  = "bottom")

ggsave("outputs/Plot5_Scatter_Oil_vs_lnIPI.png", plot = p5,
       width = 8, height = 5, dpi = 300)
print(p5)
cat("Saved: Plot5_Scatter_Oil_vs_lnIPI.png\n")


# =============================================================================
# PART 12 — SESSION INFO (Reproducibility)
# =============================================================================

cat("\n===== SESSION INFO =====\n")
sessionInfo()

# Close output log
sink()

cat("\n============================================================\n")
cat("SCRIPT COMPLETE — Files produced:\n")
cat("  Logs    : GROUP_Results_Log.txt\n")
cat("  Tables  : Table0_Descriptive_Stats.txt\n")
cat("            Table1_CPI_Models.html\n")
cat("            Table2_IPI_FE_Model.txt\n")
cat("            Table2b_IPI_HC3_vs_NWHAC.txt\n")
cat("            Table3_Asymmetry_CPI_IPI.txt\n")
cat("            Table4_Country_CPI_Canada_vs_USA.txt\n")
cat("            Table5_Country_IPI_Canada_vs_USA.txt\n")
cat("  Plots   : Plot1_Brent_Oil_Price.png\n")
cat("            Plot2_ln_CPI.png\n")
cat("            Plot3_ln_IPI.png\n")
cat("            Plot4_Scatter_Oil_vs_lnCPI.png\n")
cat("            Plot5_Scatter_Oil_vs_lnIPI.png\n")
cat("  NOTES   : RE model attempted (Part 5C) — failed, N=2 countries\n")
cat("            Hausman test attempted (Part 6B) — failed, RE=NULL\n")
cat("            Importers/Exporters split attempted (Part 10A) — failed,\n")
cat("            all obs are exporters; Canada vs USA used as substitute\n")
cat("============================================================\n")
