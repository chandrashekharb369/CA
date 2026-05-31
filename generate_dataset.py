"""
Phase 1: Synthetic Financial Dataset Generator — Schedule III (Companies Act 2013)
Generates 12,000+ rows of realistic company ledger data aligned with
Schedule III format, proper GST calculations, and controlled noise.
"""

import pandas as pd
import numpy as np
import random
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MISSING_RATE = 0.015   # ~1.5% missing values

# Category distribution defaults
COMPANY_PROFILES = {
    "retail": {
        "CATEGORY_DIST": {"Expense": 0.50, "Income": 0.30, "Asset": 0.10, "Liability": 0.10},
    },
    "generic": {
        "CATEGORY_DIST": {"Expense": 0.44, "Income": 0.24, "Asset": 0.16, "Liability": 0.16},
    },
    "trading": {
        "CATEGORY_DIST": {"Expense": 0.50, "Income": 0.30, "Asset": 0.10, "Liability": 0.10},
    },
    "manufacturing": {
        "CATEGORY_DIST": {"Expense": 0.40, "Income": 0.20, "Asset": 0.25, "Liability": 0.15},
    },
    "service": {
        "CATEGORY_DIST": {"Expense": 0.35, "Income": 0.45, "Asset": 0.10, "Liability": 0.10},
    },
    "startup": {
        "CATEGORY_DIST": {"Expense": 0.60, "Income": 0.10, "Asset": 0.10, "Liability": 0.20},
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Schedule III Sub-Category Map
# Each sub_category maps to a Schedule III head via SCHEDULE_III_HEAD_MAP below
# ─────────────────────────────────────────────────────────────────────────────
SUBCATEGORY_MAP = {
    # ── P&L — EXPENSE ────────────────────────────────────────────────────────
    "Expense": [
        # Cost of Material Consumed & Purchases
        "Raw Material Consumed",
        "Packing Material Consumed",
        "Purchases of Stock-in-Trade",
        # Employee Benefit Expenses
        "Salary & Wages",
        "Bonus & Incentives",
        "Gratuity Expense",
        "Staff Welfare",
        # Finance Costs
        "Bank Interest",
        "Loan Interest",
        "Bank Charges & Commission",
        # Depreciation & Amortisation
        "Depreciation",
        "Amortisation of Intangibles",
        # Other Expenses
        "Rent",
        "Power & Fuel",
        "Insurance Premium",
        "Advertising & Marketing",
        "Travel & Conveyance",
        "Repairs & Maintenance",
        "Legal & Professional Fees",
        "Audit Fees",
        "Telephone & Internet",
        "Printing & Stationery",
        "Miscellaneous Expense",
    ],
    # ── P&L — INCOME ─────────────────────────────────────────────────────────
    "Income": [
        # Revenue from Operations
        "Product Sales",
        "Service Revenue",
        "Export Revenue",
        # Other Income
        "Interest Income",
        "Dividend Income",
        "Rental Income",
        "Commission Income",
        "Gain on Sale of Asset",
        "Foreign Exchange Gain",
    ],
    # ── BS — ASSETS ──────────────────────────────────────────────────────────
    "Asset": [
        # Non-current — Fixed Assets (Tangible)
        "PPE - Land & Building",
        "PPE - Plant & Machinery",
        "PPE - Furniture & Fixtures",
        "PPE - Vehicles",
        "PPE - Office Equipment",
        # Non-current — Fixed Assets (Intangible)
        "Intangible - Goodwill",
        "Intangible - Patents & Trademarks",
        "Intangible - Computer Software",
        # Non-current — Other
        "Capital Work-in-Progress",
        "Non-current Investments",
        "Deferred Tax Asset",
        "Long-term Loans & Advances",
        # Current Assets
        "Inventories",
        "Trade Receivables",
        "Cash & Cash Equivalents",
        "Short-term Loans & Advances",
        "Prepaid Expenses",
        "Advance Tax & TDS Receivable",
    ],
    # ── BS — LIABILITIES (incl. Equity) ──────────────────────────────────────
    "Liability": [
        # Shareholders' Funds
        "Share Capital",
        "General Reserve",
        "Securities Premium",
        "Retained Earnings / Surplus",
        # Non-current Liabilities
        "Long-term Borrowings - Term Loan",
        "Debentures",
        "Deferred Tax Liability",
        "Lease Obligation - Long-term",
        "Security Deposit Received",
        "Provision for Gratuity",
        "Leave Encashment Provision",
        # Current Liabilities
        "Bank Overdraft",
        "Cash Credit Limit",
        "Trade Payables",
        "Advance Received from Customers",
        "Unearned Revenue",
        "Salary & Wages Payable",
        "Provision for Income Tax",
        "Proposed Dividend",
        "Other Current Liabilities",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Schedule III Head map  →  exact line item name as it appears in Schedule III
# ─────────────────────────────────────────────────────────────────────────────
SCHEDULE_III_HEAD_MAP = {
    # EXPENSE → P&L
    "Raw Material Consumed":         "Cost of Material Consumed",
    "Packing Material Consumed":     "Cost of Material Consumed",
    "Purchases of Stock-in-Trade":   "Purchases of Stock-in-Trade",
    "Salary & Wages":                "Employee Benefit Expenses",
    "Bonus & Incentives":            "Employee Benefit Expenses",
    "Gratuity Expense":              "Employee Benefit Expenses",
    "Staff Welfare":                 "Employee Benefit Expenses",
    "Bank Interest":                 "Finance Costs",
    "Loan Interest":                 "Finance Costs",
    "Bank Charges & Commission":     "Finance Costs",
    "Depreciation":                  "Depreciation & Amortisation",
    "Amortisation of Intangibles":   "Depreciation & Amortisation",
    "Rent":                          "Other Expenses",
    "Power & Fuel":                  "Other Expenses",
    "Insurance Premium":             "Other Expenses",
    "Advertising & Marketing":       "Other Expenses",
    "Travel & Conveyance":           "Other Expenses",
    "Repairs & Maintenance":         "Other Expenses",
    "Legal & Professional Fees":     "Other Expenses",
    "Audit Fees":                    "Other Expenses",
    "Telephone & Internet":          "Other Expenses",
    "Printing & Stationery":         "Other Expenses",
    "Miscellaneous Expense":         "Other Expenses",

    # INCOME → P&L
    "Product Sales":                 "Revenue from Operations",
    "Service Revenue":               "Revenue from Operations",
    "Export Revenue":                "Revenue from Operations",
    "Interest Income":               "Other Income",
    "Dividend Income":               "Other Income",
    "Rental Income":                 "Other Income",
    "Commission Income":             "Other Income",
    "Gain on Sale of Asset":         "Other Income",
    "Foreign Exchange Gain":         "Other Income",

    # ASSET → Balance Sheet
    "PPE - Land & Building":             "Tangible Fixed Assets",
    "PPE - Plant & Machinery":           "Tangible Fixed Assets",
    "PPE - Furniture & Fixtures":        "Tangible Fixed Assets",
    "PPE - Vehicles":                    "Tangible Fixed Assets",
    "PPE - Office Equipment":            "Tangible Fixed Assets",
    "Intangible - Goodwill":             "Intangible Fixed Assets",
    "Intangible - Patents & Trademarks": "Intangible Fixed Assets",
    "Intangible - Computer Software":    "Intangible Fixed Assets",
    "Capital Work-in-Progress":          "Capital Work-in-Progress",
    "Non-current Investments":           "Non-current Investments",
    "Deferred Tax Asset":                "Deferred Tax Assets (Net)",
    "Long-term Loans & Advances":        "Long-term Loans & Advances",
    "Inventories":                       "Inventories",
    "Trade Receivables":                 "Trade Receivables",
    "Cash & Cash Equivalents":           "Cash & Cash Equivalents",
    "Short-term Loans & Advances":       "Short-term Loans & Advances",
    "Prepaid Expenses":                  "Other Current Assets",
    "Advance Tax & TDS Receivable":      "Other Current Assets",

    # LIABILITY → Balance Sheet
    "Share Capital":                     "Share Capital",
    "General Reserve":                   "Reserves & Surplus",
    "Securities Premium":                "Reserves & Surplus",
    "Retained Earnings / Surplus":       "Reserves & Surplus",
    "Long-term Borrowings - Term Loan":  "Long-term Borrowings",
    "Debentures":                        "Long-term Borrowings",
    "Deferred Tax Liability":            "Deferred Tax Liabilities (Net)",
    "Lease Obligation - Long-term":      "Other Long-term Liabilities",
    "Security Deposit Received":         "Other Long-term Liabilities",
    "Provision for Gratuity":            "Long-term Provisions",
    "Leave Encashment Provision":        "Long-term Provisions",
    "Bank Overdraft":                    "Short-term Borrowings",
    "Cash Credit Limit":                 "Short-term Borrowings",
    "Trade Payables":                    "Trade Payables",
    "Advance Received from Customers":   "Other Current Liabilities",
    "Unearned Revenue":                  "Other Current Liabilities",
    "Salary & Wages Payable":            "Other Current Liabilities",
    "Provision for Income Tax":          "Short-term Provisions",
    "Proposed Dividend":                 "Short-term Provisions",
    "Other Current Liabilities":         "Other Current Liabilities",
}

# ─────────────────────────────────────────────────────────────────────────────
# GST rates — realistic Schedule III applicability
# ─────────────────────────────────────────────────────────────────────────────
GST_RATES = [0, 5, 12, 18, 28]

# Sub-category-level GST applicability (overrides category default)
GST_EXEMPT_SUBCATS = {
    # These are exempt / zero-rated
    "Share Capital", "General Reserve", "Securities Premium",
    "Retained Earnings / Surplus", "Long-term Borrowings - Term Loan",
    "Debentures", "Bank Overdraft", "Cash Credit Limit",
    "Provision for Income Tax", "Proposed Dividend",
    "Deferred Tax Liability", "Deferred Tax Asset",
    "PPE - Land & Building", "Capital Work-in-Progress",
    "Non-current Investments", "Trade Receivables",
    "Cash & Cash Equivalents", "Inventories",
    "Salary & Wages", "Bonus & Incentives", "Gratuity Expense",
    "Staff Welfare", "Bank Interest", "Loan Interest",
    "Depreciation", "Amortisation of Intangibles",
    "Interest Income", "Dividend Income", "Gain on Sale of Asset",
    "Foreign Exchange Gain",
}

GST_WEIGHTS = {
    "Expense": [0.20, 0.10, 0.20, 0.40, 0.10],
    "Income":  [0.20, 0.10, 0.20, 0.40, 0.10],
    "Asset":   [1.0, 0.0, 0.0, 0.0, 0.0],
    "Liability": [1.0, 0.0, 0.0, 0.0, 0.0],
}

PAYMENT_MODES = ["Cash", "UPI", "Bank Transfer", "NEFT/RTGS", "Cheque", "Credit"]
PAYMENT_WEIGHTS = [0.10, 0.15, 0.30, 0.20, 0.10, 0.15]

# ─────────────────────────────────────────────────────────────────────────────
# Amount ranges per sub-category (realistic Indian company values in INR)
# ─────────────────────────────────────────────────────────────────────────────
AMOUNT_RANGES = {
    # High-value capital items
    "PPE - Land & Building":             (500_000, 5_000_000),
    "PPE - Plant & Machinery":           (200_000, 2_000_000),
    "PPE - Vehicles":                    (300_000, 1_500_000),
    "PPE - Furniture & Fixtures":        (50_000,  500_000),
    "PPE - Office Equipment":            (30_000,  300_000),
    "Intangible - Goodwill":             (500_000, 3_000_000),
    "Intangible - Patents & Trademarks": (100_000, 800_000),
    "Intangible - Computer Software":    (50_000,  500_000),
    "Capital Work-in-Progress":          (200_000, 2_000_000),
    "Non-current Investments":           (100_000, 1_000_000),
    "Long-term Borrowings - Term Loan":  (500_000, 5_000_000),
    "Debentures":                        (500_000, 3_000_000),
    "Share Capital":                     (500_000, 5_000_000),
    # Mid-value items
    "Product Sales":                     (50_000,  500_000),
    "Service Revenue":                   (30_000,  300_000),
    "Export Revenue":                    (100_000, 800_000),
    "Salary & Wages":                    (30_000,  200_000),
    "Trade Receivables":                 (50_000,  400_000),
    "Trade Payables":                    (50_000,  400_000),
    "Inventories":                       (50_000,  600_000),
    "Cash & Cash Equivalents":           (10_000,  300_000),
    "Bank Overdraft":                    (50_000,  500_000),
    "General Reserve":                   (100_000, 1_000_000),
    "Securities Premium":                (200_000, 1_000_000),
    "Retained Earnings / Surplus":       (100_000, 800_000),
    # Operational expense items
    "Raw Material Consumed":             (20_000,  200_000),
    "Purchases of Stock-in-Trade":       (30_000,  400_000),
    "Rent":                              (15_000,  80_000),
    "Advertising & Marketing":           (10_000,  100_000),
    "Bank Interest":                     (5_000,   50_000),
    "Loan Interest":                     (10_000,  100_000),
    "Provision for Income Tax":          (50_000,  300_000),
}
DEFAULT_AMOUNT_RANGE = (5_000, 100_000)

# ─────────────────────────────────────────────────────────────────────────────
# Vendor / Client name pools
# ─────────────────────────────────────────────────────────────────────────────
VENDORS = [
    "Reliance Industries Ltd", "Infosys Ltd", "Tata Consultancy Services",
    "HDFC Bank Ltd", "State Bank of India", "Wipro Limited",
    "HCL Technologies", "L&T Finance Holdings", "Mahindra & Mahindra",
    "Bajaj Auto Ltd", "Aditya Birla Capital", "Godrej Industries",
    "Tech Mahindra", "Sun Pharmaceutical", "Dr. Reddy's Laboratories",
    "Cipla Ltd", "ITC Limited", "Hindustan Unilever Ltd",
    "Asian Paints Ltd", "Maruti Suzuki India",
    "M/s Sharma & Sons", "M/s Patel Enterprises Pvt. Ltd.",
    "Kumar Associates", "Gupta Trading Co.", "Mehta Consultants LLP",
    "Singh & Brothers", "Jain Hardware Pvt. Ltd.",
    "Verma Constructions", "Reddy IT Solutions",
    "Nair Technologies India", "Pillai Services Pvt. Ltd.",
    "Mishra & Co.", "Agarwal Logistics",
    "Amazon Business India", "Flipkart B2B Commerce",
    "Havells India Ltd", "Voltas Ltd", "Samsung India Pvt. Ltd.",
    "LG Electronics India", "Tata Power Ltd",
    "ICICI Bank Ltd", "Axis Bank Ltd", "Kotak Mahindra Bank",
]

# ─────────────────────────────────────────────────────────────────────────────
# Description templates (Schedule III sub-category aware)
# ─────────────────────────────────────────────────────────────────────────────
DESCRIPTION_TEMPLATES = {
    "Raw Material Consumed":         ["Raw material usage", "Material consumed for production", "RM consumed - {month}"],
    "Packing Material Consumed":     ["Packing material used", "Packaging consumed", "PM consumed - {month}"],
    "Purchases of Stock-in-Trade":   ["Purchase of trading goods", "Stock purchase - {month}", "Goods purchased"],
    "Salary & Wages":                ["Salary payment - {month}", "Monthly payroll disbursement", "Wages paid - {month}", "Staff salary {month}"],
    "Bonus & Incentives":            ["Performance bonus payment", "Annual incentive disbursement", "Quarterly bonus - {month}"],
    "Gratuity Expense":              ["Gratuity provision for {month}", "Gratuity contribution", "Gratuity expense - FY"],
    "Staff Welfare":                 ["Staff welfare expense", "Employee welfare - {month}", "Canteen & welfare charges"],
    "Bank Interest":                 ["Bank interest charges - {month}", "Interest on CC limit", "OD interest"],
    "Loan Interest":                 ["Term loan interest - {month}", "Interest on term loan", "Loan interest paid"],
    "Bank Charges & Commission":     ["Bank charges", "DD/NEFT charges", "Bank commission - {month}"],
    "Depreciation":                  ["Depreciation for {month}", "Depreciation - Plant & Machinery", "Monthly depreciation"],
    "Amortisation of Intangibles":   ["Amortisation of software", "Intangible amortisation - {month}"],
    "Rent":                          ["Office rent - {month}", "Monthly rent payment", "Lease rent - {month}"],
    "Power & Fuel":                  ["Electricity bill - {month}", "Power charges {month}", "Fuel & energy charges"],
    "Insurance Premium":             ["Annual fire insurance premium", "Health insurance - {month}", "General insurance renewal"],
    "Advertising & Marketing":       ["Digital marketing campaign", "Ad spend - {month}", "Brand promotion charges"],
    "Travel & Conveyance":           ["Business travel expense", "Conveyance charges - {month}", "Client visit expense"],
    "Repairs & Maintenance":         ["AMC payment", "Equipment repair charges", "Maintenance contract - {month}"],
    "Legal & Professional Fees":     ["Legal retainer fee - {month}", "Advocate charges", "Professional advisory"],
    "Audit Fees":                    ["Statutory audit fees", "Internal audit fees", "Tax audit charges"],
    "Telephone & Internet":          ["Mobile & broadband - {month}", "Telecom bill {month}", "Internet charges"],
    "Printing & Stationery":         ["Stationery purchase", "Printer consumables", "Office printing charges"],
    "Miscellaneous Expense":         ["Sundry expense - {month}", "Misc. charges", "Petty cash expense"],
    "Product Sales":                 ["Product sales - {month}", "Goods sold to customer", "Sales invoice - {month}"],
    "Service Revenue":               ["Service charges billed", "Consulting services rendered", "Professional fee billed"],
    "Export Revenue":                ["Export invoice", "Foreign currency sales", "Export proceeds received"],
    "Interest Income":               ["Interest on FD", "Bank deposit interest received", "FD interest - {month}"],
    "Dividend Income":               ["Dividend received", "Dividend income - {month}"],
    "Rental Income":                 ["Rental income from property", "Rent received - {month}"],
    "Commission Income":             ["Commission earned", "Agency commission received", "Referral commission"],
    "Gain on Sale of Asset":         ["Profit on asset disposal", "Gain on sale of vehicle", "Asset sale proceeds"],
    "Foreign Exchange Gain":         ["Forex gain on export", "Exchange rate gain"],
    "PPE - Land & Building":         ["Purchase of land & building", "Property acquisition", "Building construction payment"],
    "PPE - Plant & Machinery":       ["Plant & machinery purchase", "New machinery acquisition", "Equipment capital purchase"],
    "PPE - Furniture & Fixtures":    ["Office furniture purchase", "Workstation procurement", "Furniture acquisition"],
    "PPE - Vehicles":                ["Company vehicle purchase", "Delivery van acquisition", "Vehicle asset purchase"],
    "PPE - Office Equipment":        ["Office equipment purchase", "Laptop & desktop procurement", "IT equipment capital"],
    "Intangible - Goodwill":         ["Goodwill on acquisition", "Business goodwill recorded"],
    "Intangible - Patents & Trademarks": ["Patent acquisition cost", "Trademark registration"],
    "Intangible - Computer Software": ["ERP software purchase", "Software license (capital)", "Licensed software acquisition"],
    "Capital Work-in-Progress":      ["CWIP - building under construction", "Capital advance to contractor", "WIP payment"],
    "Non-current Investments":       ["Equity investment in subsidiary", "Investment in mutual funds (long-term)"],
    "Deferred Tax Asset":            ["DTA recognised - timing differences", "Deferred tax asset created"],
    "Long-term Loans & Advances":    ["Security deposit paid", "Long-term advance to supplier"],
    "Inventories":                   ["Closing stock - finished goods", "Raw material stock", "WIP inventory"],
    "Trade Receivables":             ["Debtor balance - customer", "Outstanding receivable", "Trade debtor"],
    "Cash & Cash Equivalents":       ["Cash in hand", "Bank balance", "Petty cash"],
    "Short-term Loans & Advances":   ["Advance to vendor", "Advance payment to supplier", "Staff advance"],
    "Prepaid Expenses":              ["Prepaid insurance", "Advance rent paid", "Prepaid subscription"],
    "Advance Tax & TDS Receivable":  ["Advance tax paid", "TDS receivable", "Self-assessment tax"],
    "Share Capital":                 ["Equity share capital", "Paid-up capital", "Share allotment proceeds"],
    "General Reserve":               ["Transfer to general reserve", "General reserve created"],
    "Securities Premium":            ["Securities premium on issue", "Share premium account"],
    "Retained Earnings / Surplus":   ["Retained earnings b/f", "Surplus carried forward", "P&L surplus"],
    "Long-term Borrowings - Term Loan": ["Term loan from bank", "Long-term loan disbursement"],
    "Debentures":                    ["NCD issued", "Debenture proceeds received"],
    "Deferred Tax Liability":        ["DTL created - depreciation timing", "Deferred tax liability"],
    "Lease Obligation - Long-term":  ["Finance lease liability", "Long-term lease obligation"],
    "Security Deposit Received":     ["Security deposit from tenant", "Refundable deposit received"],
    "Provision for Gratuity":        ["Gratuity provision - year end", "AS 15 gratuity provision"],
    "Leave Encashment Provision":    ["Leave encashment provision", "PL encashment liability"],
    "Bank Overdraft":                ["OD account balance", "Bank overdraft outstanding"],
    "Cash Credit Limit":             ["CC account utilisation", "Cash credit outstanding"],
    "Trade Payables":                ["Creditor balance - supplier", "Trade payable - outstanding", "Vendor payable"],
    "Advance Received from Customers": ["Advance received from client", "Customer advance payment", "Booking advance"],
    "Unearned Revenue":              ["Deferred revenue", "Subscription income deferred"],
    "Salary & Wages Payable":        ["Salary payable - {month}", "Outstanding wages", "Accrued payroll"],
    "Provision for Income Tax":      ["Provision for tax - FY", "Income tax provision", "Current tax liability"],
    "Proposed Dividend":             ["Proposed equity dividend", "Final dividend declared"],
    "Other Current Liabilities":     ["Other payables", "Sundry creditors", "Miscellaneous payable"],
}

MONTHS = [
    "April", "May", "June", "July", "August", "September",
    "October", "November", "December", "January", "February", "March"
]

NOISE_WORDS = {
    "payment": ["payemnt", "paymnt"],
    "received": ["recieved", "receivd"],
    "purchase": ["purchace", "purchse"],
    "salary":   ["salry", "salery"],
    "charges":  ["charjes", "chrges"],
}


def add_spelling_noise(text: str) -> str:
    for word, variants in NOISE_WORDS.items():
        if word in text.lower() and random.random() < 0.3:
            return text.lower().replace(word, random.choice(variants))
    return text


def generate_description(sub_cat: str, apply_noise: bool = False) -> str:
    month = random.choice(MONTHS)
    templates = DESCRIPTION_TEMPLATES.get(sub_cat, [sub_cat])
    desc = random.choice(templates).replace("{month}", month)
    if apply_noise and random.random() < 0.20:
        desc = add_spelling_noise(desc)
    return desc


def random_date(start: datetime, end: datetime) -> str:
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")


def get_amount(sub_cat: str) -> float:
    lo, hi = AMOUNT_RANGES.get(sub_cat, DEFAULT_AMOUNT_RANGE)
    # Use log-normal within the range for realism
    raw = np.random.lognormal(mean=np.log((lo + hi) / 2), sigma=0.35)
    return round(max(lo * 0.5, min(hi * 1.5, raw)) / 100) * 100  # round to nearest 100


# ─────────────────────────────────────────────────────────────────────────────
# Main row generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_row(category: str, start_date: datetime, end_date: datetime, is_anomaly: bool = False, company_type: str = "generic") -> dict:
    sub_cat = random.choice(SUBCATEGORY_MAP[category])
    
    # Profile overrides
    if company_type == "service" and sub_cat in ("Raw Material Consumed", "Packing Material Consumed", "Inventories", "Capital Work-in-Progress", "Purchases of Stock-in-Trade"):
        sub_cat = "Salary & Wages" if category == "Expense" else random.choice(SUBCATEGORY_MAP[category])
    elif company_type in ("retail", "trading") and sub_cat in ("PPE - Plant & Machinery", "Raw Material Consumed", "Capital Work-in-Progress"):
        sub_cat = "Purchases of Stock-in-Trade" if category == "Expense" else ("Product Sales" if category == "Income" else random.choice(SUBCATEGORY_MAP[category]))

    amount = get_amount(sub_cat)

    tx_type = "Debit" if category in ("Expense", "Asset") else "Credit"

    # GST logic: exempt sub-categories always get 0%
    if sub_cat in GST_EXEMPT_SUBCATS:
        gst_pct = 0
    else:
        gst_pct = random.choices(GST_RATES, weights=GST_WEIGHTS[category])[0]

    pay_mode = random.choices(PAYMENT_MODES, weights=PAYMENT_WEIGHTS)[0]
    vendor   = random.choice(VENDORS)
    date     = random_date(start_date, end_date)
    
    # Inject Anomalies
    if is_anomaly:
        anomaly_type = random.choice(["huge_amount", "cash_limit", "wrong_gst", "weekend_night"])
        if anomaly_type == "huge_amount":
            amount *= random.uniform(20.0, 100.0)
        elif anomaly_type == "cash_limit":
            amount = random.uniform(500_000, 2_000_000)
            pay_mode = "Cash"
        elif anomaly_type == "wrong_gst":
            gst_pct = 18 if gst_pct == 0 else 0

    gst_amt = round(amount * gst_pct / 100, 2) if gst_pct > 0 else 0.0

    apply_noise = random.random() < 0.18
    description = generate_description(sub_cat, apply_noise=apply_noise)

    s3_head = SCHEDULE_III_HEAD_MAP.get(sub_cat, sub_cat)

    return {
        "Date":               date,
        "Description":        description,
        "Amount":             amount,
        "Transaction_Type":   tx_type,
        "Category":           category,
        "Sub_Category":       sub_cat,
        "Schedule_III_Head":  s3_head,
        "GST_Percentage":     gst_pct,
        "GST_Amount":         gst_amt,
        "Payment_Mode":       pay_mode,
        "Vendor_Client_Name": vendor,
        "Is_Anomaly":         is_anomaly,
    }


def apply_missing_values(df: pd.DataFrame, rate: float = MISSING_RATE) -> pd.DataFrame:
    """Inject NaN into ~rate fraction of cells (except key accounting fields)."""
    nullable_cols = ["Description", "Vendor_Client_Name", "Payment_Mode"]
    n_missing = int(len(df) * rate)
    for _ in range(n_missing):
        col = random.choice(nullable_cols)
        idx = random.randint(0, len(df) - 1)
        df.at[idx, col] = np.nan
    return df


def generate_dataset(output_path: str = "train_dataset/financial_dataset.csv", anomaly_rate: float = 0.05) -> pd.DataFrame:
    # 10 diverse companies + 50 similar companies (Retail) => 60 companies total
    num_similar = 50
    num_diverse = 10
    total_companies = num_similar + num_diverse
    
    print(f"[Phase 1] Generating dataset for {total_companies} companies (Anomaly Rate={anomaly_rate*100:.1f}%)...")
    
    all_rows = []
    
    for comp_idx in range(1, total_companies + 1):
        if comp_idx <= num_similar:
            company_type = "retail"
        else:
            company_type = random.choice(["manufacturing", "service", "startup"])
            
        company_id = f"COMP_{comp_idx:03d}"
        industry = company_type.capitalize()
        # Entries per company: 2,000 - 10,000
        n_rows = random.randint(2000, 10000)
        
        # Determine ledger duration: randomly create 5 year old ledger for some
        # We'll make about 20% of companies have a full 5-year history.
        # Others will have 1-year history but spread across the last 5 years.
        is_5_year = random.random() < 0.20
        if is_5_year:
            start_date = datetime(2019, 4, 1)
            end_date = datetime(2024, 3, 31)
        else:
            year = random.randint(2019, 2023)
            start_date = datetime(year, 4, 1)
            end_date = datetime(year + 1, 3, 31)
            
        cat_dist = COMPANY_PROFILES.get(company_type, COMPANY_PROFILES["generic"])["CATEGORY_DIST"]
        categories = random.choices(list(cat_dist.keys()), weights=list(cat_dist.values()), k=n_rows)
        
        for cat in categories:
            is_anomaly = random.random() < anomaly_rate
            row_data = generate_row(cat, start_date, end_date, is_anomaly=is_anomaly, company_type=company_type)
            row_data["Company_ID"] = company_id
            row_data["Company_Name"] = f"Company {comp_idx} Pvt Ltd"
            row_data["Industry"] = industry
            all_rows.append(row_data)

    df = pd.DataFrame(all_rows)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Company_ID", "Date"]).reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    df = apply_missing_values(df)
    
    # Reorder columns so Company info is first
    cols = ["Company_ID", "Company_Name", "Industry", "Date", "Description", "Amount", "Transaction_Type", 
            "Category", "Sub_Category", "Schedule_III_Head", "GST_Percentage", "GST_Amount", 
            "Payment_Mode", "Vendor_Client_Name", "Is_Anomaly"]
    df = df[cols]

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the combined dataset (useful for training/pipeline)
    df.to_csv(output_path, index=False)
    print(f"  Combined dataset saved -> {output_path}  ({len(df):,} rows)")
    
    # Save individual csv for each company
    companies_dir = os.path.join(os.path.dirname(output_path), "companies")
    os.makedirs(companies_dir, exist_ok=True)
    for company_id, group_df in df.groupby("Company_ID"):
        comp_file = os.path.join(companies_dir, f"{company_id}.csv")
        group_df.to_csv(comp_file, index=False)
    print(f"  Individual company CSV files saved in -> {companies_dir}")

    print(f"\n  Total Companies Generated: {len(df['Company_ID'].unique())}")
    print("\n  Category distribution:")
    dist = df["Category"].value_counts(normalize=True) * 100
    for cat, pct in dist.items():
        print(f"    {cat:<14}: {pct:.1f}%")

    missing_total = df.isnull().sum().sum()
    print(f"\n  Missing values injected: {missing_total} "
          f"({missing_total / (len(df) * len(df.columns)) * 100:.2f}%)")

    return df


if __name__ == "__main__":
    generate_dataset()
