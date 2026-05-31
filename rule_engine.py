"""
Phase 4: Backward Chaining Rule Engine — Schedule III (Companies Act 2013)
Computes all Schedule III line items via backward chaining inference,
producing a FinancialSummary suitable for generating a full Schedule III
Balance Sheet and Statement of Profit & Loss.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd
import numpy as np


# Knowledge Base — facts loaded from the dataset
@dataclass
class FinancialFact:
    """A single accounting transaction as a fact in the knowledge base."""
    row_id:           int
    date:             str
    description:      str
    amount:           float
    category:         str          # Predicted or actual main category
    sub_category:     str          # Schedule III sub-category
    schedule_iii_head: str         # Exact Schedule III line item
    gst_percentage:   float
    gst_amount:       float
    payment_mode:     str
    vendor:           str


@dataclass
class FinancialSummary:
    """
    Full Schedule III result produced by the backward chaining engine.

    Balance Sheet — I. EQUITY AND LIABILITIES
      (1) Shareholders' Funds
      (2) Non-Current Liabilities
      (3) Current Liabilities

    Balance Sheet — II. ASSETS
      Non-Current Assets (Fixed + Other)
      Current Assets

    P&L Statement
      Revenue from Operations + Other Income
      Expenses by head
    """

    # ── P&L ──────────────────────────────────────────────────────────────────
    revenue_from_operations: float = 0.0
    other_income:            float = 0.0

    cost_of_materials:       float = 0.0   # Raw + Packing material consumed
    employee_benefit_expense: float = 0.0  # Salary, Bonus, Gratuity, Welfare
    finance_costs:           float = 0.0   # Bank interest, Loan interest, charges
    depreciation_amortisation: float = 0.0 # Depreciation + Amortisation
    other_expenses:          float = 0.0   # Rent, Power, Insurance, Misc...

    # GST
    gst_on_income:           float = 0.0   # Output tax
    gst_on_expense:          float = 0.0   # Input tax credit

    # ── Balance Sheet — I. EQUITY AND LIABILITIES ────────────────────────────
    # (1) Shareholders' Funds
    share_capital:           float = 0.0
    reserves_surplus:        float = 0.0

    # (2) Non-Current Liabilities
    long_term_borrowings:    float = 0.0
    deferred_tax_liability:  float = 0.0
    other_lt_liabilities:    float = 0.0   # Lease+Security deposits
    lt_provisions:           float = 0.0   # Gratuity, Leave encashment provision

    # (3) Current Liabilities
    short_term_borrowings:   float = 0.0   # OD, CC limit
    trade_payables:          float = 0.0
    other_current_liabilities: float = 0.0 # Advances received, Unearned, Salary payable
    short_term_provisions:   float = 0.0   # Tax provision, Proposed dividend

    # ── Balance Sheet — II. ASSETS ───────────────────────────────────────────
    # Non-current — Fixed Assets
    tangible_assets:         float = 0.0   # PPE: Land, Plant, Vehicles, etc.
    intangible_assets:       float = 0.0   # Goodwill, Patents, Software
    cwip:                    float = 0.0   # Capital Work-in-Progress

    # Non-current — Other
    non_current_investments: float = 0.0
    deferred_tax_asset:      float = 0.0
    lt_loans_advances:       float = 0.0   # Long-term loans & advances

    # Current Assets
    inventories:             float = 0.0
    trade_receivables:       float = 0.0
    cash_equivalents:        float = 0.0
    short_term_loans:        float = 0.0
    other_current_assets:    float = 0.0   # Prepaid, Advance Tax

    # ── Raw transaction buckets (for reporting) ───────────────────────────────
    income_txns:    List[FinancialFact] = field(default_factory=list)
    expense_txns:   List[FinancialFact] = field(default_factory=list)
    asset_txns:     List[FinancialFact] = field(default_factory=list)
    liability_txns: List[FinancialFact] = field(default_factory=list)

    # ── Derived totals ────────────────────────────────────────────────────────
    # Diagnostic / balancing
    balance_sheet_adjustment: float = 0.0  # Applied to Reserves & Surplus to enforce A = E+L
    ca_suggestions: List[str] = field(default_factory=list)  # Actionable CA insights

    @property
    def total_income(self) -> float:
        return self.revenue_from_operations + self.other_income

    @property
    def total_expense(self) -> float:
        return (self.cost_of_materials + self.employee_benefit_expense
                + self.finance_costs + self.depreciation_amortisation
                + self.other_expenses)

    @property
    def profit_before_tax(self) -> float:
        return self.total_income - self.total_expense

    @property
    def net_profit_loss(self) -> float:
        """After providing for income tax (short-term provisions used as proxy)."""
        return self.profit_before_tax - self.short_term_provisions

    @property
    def profit_margin_pct(self) -> float:
        if self.total_income == 0:
            return 0.0
        return (self.profit_before_tax / self.total_income) * 100

    @property
    def gst_payable(self) -> float:
        return max(0.0, self.gst_on_income - self.gst_on_expense)

    # ── BS sub-totals ─────────────────────────────────────────────────────────
    @property
    def shareholders_funds(self) -> float:
        return self.share_capital + self.reserves_surplus

    @property
    def non_current_liabilities(self) -> float:
        return (self.long_term_borrowings + self.deferred_tax_liability
                + self.other_lt_liabilities + self.lt_provisions)

    @property
    def current_liabilities(self) -> float:
        return (self.short_term_borrowings + self.trade_payables
                + self.other_current_liabilities + self.short_term_provisions)

    @property
    def total_equity_and_liabilities(self) -> float:
        return (self.shareholders_funds + self.non_current_liabilities
                + self.current_liabilities)

    @property
    def fixed_assets_total(self) -> float:
        return self.tangible_assets + self.intangible_assets + self.cwip

    @property
    def non_current_assets(self) -> float:
        return (self.fixed_assets_total + self.non_current_investments
                + self.deferred_tax_asset + self.lt_loans_advances)

    @property
    def current_assets(self) -> float:
        return (self.inventories + self.trade_receivables + self.cash_equivalents
                + self.short_term_loans + self.other_current_assets)

    @property
    def total_assets(self) -> float:
        return self.non_current_assets + self.current_assets

    @property
    def total_liabilities(self) -> float:
        """Legacy compat — all obligations excluding equity."""
        return self.non_current_liabilities + self.current_liabilities

    def to_dict(self) -> Dict:
        return {
            "Total Income":               round(self.total_income, 2),
            "Total Expense":              round(self.total_expense, 2),
            "Net Profit / Loss":          round(self.net_profit_loss, 2),
            "Profit Before Tax":          round(self.profit_before_tax, 2),
            "Total Assets":               round(self.total_assets, 2),
            "Total Liabilities":          round(self.total_liabilities, 2),
            "Shareholders Funds":         round(self.shareholders_funds, 2),
            "GST Collected (Output)":     round(self.gst_on_income, 2),
            "GST Paid (Input Credit)":    round(self.gst_on_expense, 2),
            "Net GST Payable":            round(self.gst_payable, 2),
            "Balance Sheet Adjustment":   round(self.balance_sheet_adjustment, 2),
            "Profit Margin %":            round(self.profit_margin_pct, 2),
            "CA Suggestions":             self.ca_suggestions,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Schedule III Head → FinancialSummary field mapping
# ─────────────────────────────────────────────────────────────────────────────
# Maps each Schedule_III_Head value → which FinancialSummary attribute it adds to
S3_HEAD_FIELD_MAP = {
    # P&L Income
    "Revenue from Operations":       "revenue_from_operations",
    "Other Income":                  "other_income",

    # P&L Expense
    "Cost of Material Consumed":     "cost_of_materials",
    "Employee Benefit Expenses":     "employee_benefit_expense",
    "Finance Costs":                 "finance_costs",
    "Depreciation & Amortisation":   "depreciation_amortisation",
    "Other Expenses":                "other_expenses",

    # BS Assets — Non-current
    "Tangible Fixed Assets":         "tangible_assets",
    "Intangible Fixed Assets":       "intangible_assets",
    "Capital Work-in-Progress":      "cwip",
    "Non-current Investments":       "non_current_investments",
    "Deferred Tax Assets (Net)":     "deferred_tax_asset",
    "Long-term Loans & Advances":    "lt_loans_advances",

    # BS Assets — Current
    "Inventories":                   "inventories",
    "Trade Receivables":             "trade_receivables",
    "Cash & Cash Equivalents":       "cash_equivalents",
    "Short-term Loans & Advances":   "short_term_loans",
    "Other Current Assets":          "other_current_assets",

    # BS Liabilities — Shareholders' Funds
    "Share Capital":                 "share_capital",
    "Reserves & Surplus":            "reserves_surplus",

    # BS Liabilities — Non-current
    "Long-term Borrowings":          "long_term_borrowings",
    "Deferred Tax Liabilities (Net)":"deferred_tax_liability",
    "Other Long-term Liabilities":   "other_lt_liabilities",
    "Long-term Provisions":          "lt_provisions",

    # BS Liabilities — Current
    "Short-term Borrowings":         "short_term_borrowings",
    "Trade Payables":                "trade_payables",
    "Other Current Liabilities":     "other_current_liabilities",
    "Short-term Provisions":         "short_term_provisions",
}


# ─────────────────────────────────────────────────────────────────────────────
# Backward Chaining Rule Engine
# ─────────────────────────────────────────────────────────────────────────────
class BackwardChainingRuleEngine:
    """
    Backward chaining from top-level financial goals to leaf-level facts.

    Goal hierarchy:
        compute_schedule_iii_pl
            ├── compute_revenue_from_operations
            ├── compute_other_income
            ├── compute_cost_of_materials
            ├── compute_employee_benefit_expense
            ├── compute_finance_costs
            ├── compute_depreciation_amortisation
            └── compute_other_expenses

        compute_schedule_iii_balance_sheet
            ├── compute_shareholders_funds
            │       ├── compute_share_capital
            │       └── compute_reserves_surplus
            ├── compute_non_current_liabilities
            │       ├── compute_long_term_borrowings
            │       ├── compute_deferred_tax_liability
            │       ├── compute_other_lt_liabilities
            │       └── compute_lt_provisions
            ├── compute_current_liabilities
            │       ├── compute_short_term_borrowings
            │       ├── compute_trade_payables
            │       ├── compute_other_current_liabilities
            │       └── compute_short_term_provisions
            ├── compute_non_current_assets
            │       ├── compute_tangible_assets
            │       ├── compute_intangible_assets
            │       ├── compute_cwip
            │       ├── compute_non_current_investments
            │       ├── compute_deferred_tax_asset
            │       └── compute_lt_loans_advances
            └── compute_current_assets
                    ├── compute_inventories
                    ├── compute_trade_receivables
                    ├── compute_cash_equivalents
                    ├── compute_short_term_loans
                    └── compute_other_current_assets

        compute_gst_payable
    """

    def __init__(self, facts: List[FinancialFact]):
        self._facts = facts
        self._summary = FinancialSummary()
        self._proved: Dict[str, bool] = {}

    def run(self) -> FinancialSummary:
        print("[Phase 4] Backward Chaining Rule Engine (Schedule III) running...")
        self._prove("compute_schedule_iii_pl")
        self._prove("compute_schedule_iii_balance_sheet")
        self._prove("compute_balance_sheet_reconciliation")
        self._prove("compute_gst_payable")
        self._prove("compute_transaction_buckets")
        self._prove("compute_ca_suggestions")
        self._report()
        return self._summary

    # ── Dispatcher ────────────────────────────────────────────────────────────
    def _prove(self, goal: str) -> bool:
        if goal in self._proved:
            return self._proved[goal]
        method = getattr(self, f"_goal_{goal}", None)
        if method is None:
            self._proved[goal] = False
            return False
        result = method()
        self._proved[goal] = result
        return result

    # ── Generic leaf goal: accumulate a Schedule III head into a summary field ─
    def _accumulate(self, s3_head: str, field_name: str,
                    category_filter: str = None) -> bool:
        facts = [f for f in self._facts
                 if f.schedule_iii_head == s3_head
                 and (category_filter is None or f.category == category_filter)]
        if not facts:
            return False
        current = getattr(self._summary, field_name, 0.0)
        setattr(self._summary, field_name, current + sum(f.amount for f in facts))
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # P&L Goals
    # ──────────────────────────────────────────────────────────────────────────
    def _goal_compute_schedule_iii_pl(self) -> bool:
        results = [
            self._prove("compute_revenue_from_operations"),
            self._prove("compute_other_income"),
            self._prove("compute_cost_of_materials"),
            self._prove("compute_employee_benefit_expense"),
            self._prove("compute_finance_costs"),
            self._prove("compute_depreciation_amortisation"),
            self._prove("compute_other_expenses"),
        ]
        return any(results)

    def _goal_compute_revenue_from_operations(self) -> bool:
        return self._accumulate("Revenue from Operations", "revenue_from_operations", "Income")

    def _goal_compute_other_income(self) -> bool:
        return self._accumulate("Other Income", "other_income", "Income")

    def _goal_compute_cost_of_materials(self) -> bool:
        return self._accumulate("Cost of Material Consumed", "cost_of_materials", "Expense")

    def _goal_compute_employee_benefit_expense(self) -> bool:
        return self._accumulate("Employee Benefit Expenses", "employee_benefit_expense", "Expense")

    def _goal_compute_finance_costs(self) -> bool:
        return self._accumulate("Finance Costs", "finance_costs", "Expense")

    def _goal_compute_depreciation_amortisation(self) -> bool:
        return self._accumulate("Depreciation & Amortisation", "depreciation_amortisation", "Expense")

    def _goal_compute_other_expenses(self) -> bool:
        return self._accumulate("Other Expenses", "other_expenses", "Expense")

    # ──────────────────────────────────────────────────────────────────────────
    # Balance Sheet Goals
    # ──────────────────────────────────────────────────────────────────────────
    def _goal_compute_schedule_iii_balance_sheet(self) -> bool:
        results = [
            self._prove("compute_shareholders_funds"),
            self._prove("compute_non_current_liabilities"),
            self._prove("compute_current_liabilities"),
            self._prove("compute_non_current_assets"),
            self._prove("compute_current_assets"),
        ]
        return any(results)

    # ── Shareholders' Funds ───────────────────────────────────────────────────
    def _goal_compute_shareholders_funds(self) -> bool:
        a = self._prove("compute_share_capital")
        b = self._prove("compute_reserves_surplus")
        return a or b

    def _goal_compute_share_capital(self) -> bool:
        return self._accumulate("Share Capital", "share_capital", "Liability")

    def _goal_compute_reserves_surplus(self) -> bool:
        ok = self._accumulate("Reserves & Surplus", "reserves_surplus", "Liability")
        # Add net profit to Reserves & Surplus (retained earnings)
        npl = self._summary.profit_before_tax - self._summary.short_term_provisions
        self._summary.reserves_surplus = round(self._summary.reserves_surplus + max(0, npl), 2)
        return ok

    # ── Non-Current Liabilities ───────────────────────────────────────────────
    def _goal_compute_non_current_liabilities(self) -> bool:
        results = [
            self._prove("compute_long_term_borrowings"),
            self._prove("compute_deferred_tax_liability"),
            self._prove("compute_other_lt_liabilities"),
            self._prove("compute_lt_provisions"),
        ]
        return any(results)

    def _goal_compute_long_term_borrowings(self) -> bool:
        return self._accumulate("Long-term Borrowings", "long_term_borrowings", "Liability")

    def _goal_compute_deferred_tax_liability(self) -> bool:
        return self._accumulate("Deferred Tax Liabilities (Net)", "deferred_tax_liability", "Liability")

    def _goal_compute_other_lt_liabilities(self) -> bool:
        return self._accumulate("Other Long-term Liabilities", "other_lt_liabilities", "Liability")

    def _goal_compute_lt_provisions(self) -> bool:
        return self._accumulate("Long-term Provisions", "lt_provisions", "Liability")

    # ── Current Liabilities ───────────────────────────────────────────────────
    def _goal_compute_current_liabilities(self) -> bool:
        results = [
            self._prove("compute_short_term_borrowings"),
            self._prove("compute_trade_payables"),
            self._prove("compute_other_current_liabilities"),
            self._prove("compute_short_term_provisions"),
        ]
        return any(results)

    def _goal_compute_short_term_borrowings(self) -> bool:
        return self._accumulate("Short-term Borrowings", "short_term_borrowings", "Liability")

    def _goal_compute_trade_payables(self) -> bool:
        return self._accumulate("Trade Payables", "trade_payables", "Liability")

    def _goal_compute_other_current_liabilities(self) -> bool:
        return self._accumulate("Other Current Liabilities", "other_current_liabilities", "Liability")

    def _goal_compute_short_term_provisions(self) -> bool:
        return self._accumulate("Short-term Provisions", "short_term_provisions", "Liability")

    # ── Non-Current Assets ────────────────────────────────────────────────────
    def _goal_compute_non_current_assets(self) -> bool:
        results = [
            self._prove("compute_tangible_assets"),
            self._prove("compute_intangible_assets"),
            self._prove("compute_cwip"),
            self._prove("compute_non_current_investments"),
            self._prove("compute_deferred_tax_asset"),
            self._prove("compute_lt_loans_advances"),
        ]
        return any(results)

    def _goal_compute_tangible_assets(self) -> bool:
        return self._accumulate("Tangible Fixed Assets", "tangible_assets", "Asset")

    def _goal_compute_intangible_assets(self) -> bool:
        return self._accumulate("Intangible Fixed Assets", "intangible_assets", "Asset")

    def _goal_compute_cwip(self) -> bool:
        return self._accumulate("Capital Work-in-Progress", "cwip", "Asset")

    def _goal_compute_non_current_investments(self) -> bool:
        return self._accumulate("Non-current Investments", "non_current_investments", "Asset")

    def _goal_compute_deferred_tax_asset(self) -> bool:
        return self._accumulate("Deferred Tax Assets (Net)", "deferred_tax_asset", "Asset")

    def _goal_compute_lt_loans_advances(self) -> bool:
        return self._accumulate("Long-term Loans & Advances", "lt_loans_advances", "Asset")

    # ── Current Assets ────────────────────────────────────────────────────────
    def _goal_compute_current_assets(self) -> bool:
        results = [
            self._prove("compute_inventories"),
            self._prove("compute_trade_receivables"),
            self._prove("compute_cash_equivalents"),
            self._prove("compute_short_term_loans"),
            self._prove("compute_other_current_assets"),
        ]
        return any(results)

    def _goal_compute_inventories(self) -> bool:
        return self._accumulate("Inventories", "inventories", "Asset")

    def _goal_compute_trade_receivables(self) -> bool:
        return self._accumulate("Trade Receivables", "trade_receivables", "Asset")

    def _goal_compute_cash_equivalents(self) -> bool:
        return self._accumulate("Cash & Cash Equivalents", "cash_equivalents", "Asset")

    def _goal_compute_short_term_loans(self) -> bool:
        return self._accumulate("Short-term Loans & Advances", "short_term_loans", "Asset")

    def _goal_compute_other_current_assets(self) -> bool:
        return self._accumulate("Other Current Assets", "other_current_assets", "Asset")

    # ── GST ──────────────────────────────────────────────────────────────────
    def _goal_compute_gst_payable(self) -> bool:
        income_gst = [f for f in self._facts
                      if f.category == "Income" and f.gst_percentage > 0]
        expense_gst = [f for f in self._facts
                       if f.category == "Expense" and f.gst_percentage > 0]
        self._summary.gst_on_income  = sum(f.gst_amount for f in income_gst)
        self._summary.gst_on_expense = sum(f.gst_amount for f in expense_gst)
        return True

    # ── Balance sheet reconciliation (Schedule III presentation) ───────────
    def _goal_compute_balance_sheet_reconciliation(self) -> bool:
        """
        Enforce accounting identity for presentation:
            Total Assets = Total Equity & Liabilities.

        Synthetic data is generated independently across categories, so assets
        and liabilities may not naturally match. For reporting purposes, we
        post the balancing figure to Reserves & Surplus and track it explicitly
        in `balance_sheet_adjustment`.
        """
        gap = round(self._summary.total_assets - self._summary.total_equity_and_liabilities, 2)
        if abs(gap) < 1.0:
            self._summary.balance_sheet_adjustment = 0.0
            return True

        # Post adjustment
        self._summary.reserves_surplus += gap
        self._summary.balance_sheet_adjustment = gap

        return True
    # ── Transaction buckets (legacy compat for insights) ─────────────────────
    def _goal_compute_transaction_buckets(self) -> bool:
        self._summary.income_txns    = [f for f in self._facts if f.category == "Income"]
        self._summary.expense_txns   = [f for f in self._facts if f.category == "Expense"]
        self._summary.asset_txns     = [f for f in self._facts if f.category == "Asset"]
        self._summary.liability_txns = [f for f in self._facts if f.category == "Liability"]
        return True

    # ── CA Actionable Insights ───────────────────────────────────────────────
    def _goal_compute_ca_suggestions(self) -> bool:
        s = self._summary
        suggestions = []

        # 1. Profitability Analysis
        if s.total_income > 0:
            if s.profit_margin_pct < 0:
                suggestions.append("Critical: The business is operating at a loss. Immediate review of 'Cost of Material Consumed' and 'Other Expenses' is recommended.")
            elif s.profit_margin_pct < 5.0:
                suggestions.append(f"Low Profit Margin: Net profit margin is only {s.profit_margin_pct:.1f}%. Consider optimizing operating expenses or pricing strategies.")
            elif s.profit_margin_pct > 25.0:
                suggestions.append(f"Healthy Profitability: Strong profit margin detected ({s.profit_margin_pct:.1f}%). Consider investing surplus reserves.")
        elif s.total_expense > 0:
            suggestions.append("Critical: Expenses incurred without any operating revenue. Assess business continuity and sales strategy.")
        
        # 2. Liquidity Risk (Current Ratio)
        if s.current_liabilities > 0:
            current_ratio = s.current_assets / s.current_liabilities
            if current_ratio < 1.0:
                suggestions.append(f"Liquidity Risk: Current Ratio is {current_ratio:.2f}. The company may face difficulties covering short-term obligations using current assets.")
            elif current_ratio > 3.0:
                suggestions.append(f"Idle Asset Warning: Current Ratio is {current_ratio:.2f}. Too much capital might be tied up in current assets (like inventories or receivables).")

        # 3. Leverage (Debt to Equity)
        total_borrowings = s.long_term_borrowings + s.short_term_borrowings
        if s.shareholders_funds > 0:
            debt_to_equity = total_borrowings / s.shareholders_funds
            if debt_to_equity > 2.0:
                suggestions.append(f"High Leverage: Debt-to-Equity ratio is {debt_to_equity:.2f}. High reliance on external debt; consider raising equity or restructuring high-interest loans.")
        elif s.shareholders_funds <= 0 and total_borrowings > 0:
            suggestions.append("Critical Leverage: Company is operating on debt with negative or zero equity. Significant solvency risk.")
        
        # 4. Expenditure structure
        if s.total_expense > 0:
            finance_cost_ratio = s.finance_costs / s.total_expense
            if finance_cost_ratio > 0.15:
                suggestions.append(f"High Debt Servicing: Finance costs make up {finance_cost_ratio*100:.1f}% of total expenses. Explore refinancing options to reduce interest burden.")

        # 5. GST and Tax compliance
        if s.gst_payable > 50000:
            suggestions.append(f"High Net GST Payable (Rs.{s.gst_payable:,.2f}). Ensure all Input Tax Credit (ITC) has been properly reconciled with GSTR-2B before filing.")
            
        if s.cash_equivalents < s.gst_payable and s.gst_payable > 0:
            suggestions.append(f"Cash Crunch for Tax: Cash and equivalents (Rs.{s.cash_equivalents:,.2f}) are lower than the pending GST liability (Rs.{s.gst_payable:,.2f}). Immediate cash flow planning required.") 

        if not suggestions:
            suggestions.append("Financials appear stable. Ensure timely statutory audits and routine compliance checks.")

        s.ca_suggestions = suggestions
        return True

    # ── Console report ────────────────────────────────────────────────────────
    def _report(self):
        s = self._summary
        sep = "-" * 60

        def _pr(text=""):
            try:
                print(text)
            except UnicodeEncodeError:
                print(str(text).encode("ascii", errors="replace").decode("ascii"))

        _pr(f"\n  {sep}")
        _pr(f"  {'SCHEDULE III FINANCIAL SUMMARY':^58}")
        _pr(f"  {sep}")
        _pr("  P and L")
        _pr(f"    Revenue from Operations  : Rs.{s.revenue_from_operations:>15,.2f}")
        _pr(f"    Other Income             : Rs.{s.other_income:>15,.2f}")
        _pr(f"    Total Revenue            : Rs.{s.total_income:>15,.2f}")
        _pr(f"    Cost of Materials        : Rs.{s.cost_of_materials:>15,.2f}")
        _pr(f"    Employee Benefit Expense : Rs.{s.employee_benefit_expense:>15,.2f}")
        _pr(f"    Finance Costs            : Rs.{s.finance_costs:>15,.2f}")
        _pr(f"    Depreciation             : Rs.{s.depreciation_amortisation:>15,.2f}")
        _pr(f"    Other Expenses           : Rs.{s.other_expenses:>15,.2f}")
        _pr(f"    Profit Before Tax        : Rs.{s.profit_before_tax:>15,.2f}")
        _pr(f"  {sep}")
        _pr("  BALANCE SHEET")
        _pr(f"    Share Capital            : Rs.{s.share_capital:>15,.2f}")
        _pr(f"    Reserves and Surplus     : Rs.{s.reserves_surplus:>15,.2f}")
        _pr(f"    Long-term Borrowings     : Rs.{s.long_term_borrowings:>15,.2f}")
        _pr(f"    Current Liabilities      : Rs.{s.current_liabilities:>15,.2f}")
        _pr(f"    Total Eq and Liab        : Rs.{s.total_equity_and_liabilities:>15,.2f}")
        _pr(f"    Tangible Assets (PPE)    : Rs.{s.tangible_assets:>15,.2f}")
        _pr(f"    Current Assets           : Rs.{s.current_assets:>15,.2f}")
        _pr(f"    Total Assets             : Rs.{s.total_assets:>15,.2f}")
        _pr(f"  {sep}")

        if s.ca_suggestions:
            _pr("  CA ACTIONABLE INSIGHTS")
            for i, suggestion in enumerate(s.ca_suggestions, 1):
                _pr(f"    {i}. {suggestion}")
            _pr(f"  {sep}")

        _pr()
        proved = [g for g, r in self._proved.items() if r]
        _pr(f"  Goals proved ({len(proved)}): {proved}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: DataFrame → Facts
# ─────────────────────────────────────────────────────────────────────────────
def df_to_facts(df: pd.DataFrame, category_col: str = "Category") -> List[FinancialFact]:
    facts = []
    for idx, row in df.iterrows():
        cat = str(row.get(category_col, "")).strip()
        if cat not in ("Expense", "Income", "Asset", "Liability"):
            continue

        sub_cat  = str(row.get("Sub_Category", "")).strip()
        s3_head  = str(row.get("Schedule_III_Head", "")).strip()

        # Fallback: derive Schedule III head from sub-category if column absent
        if not s3_head or s3_head == "nan":
            from generate_dataset import SCHEDULE_III_HEAD_MAP
            s3_head = SCHEDULE_III_HEAD_MAP.get(sub_cat, sub_cat)

        facts.append(FinancialFact(
            row_id            = int(idx),
            date              = str(row.get("Date", "")),
            description       = str(row.get("Description", "")),
            amount            = float(row.get("Amount", 0)),
            category          = cat,
            sub_category      = sub_cat,
            schedule_iii_head = s3_head,
            gst_percentage    = float(row.get("GST_Percentage", 0)),
            gst_amount        = float(row.get("GST_Amount", 0)),
            payment_mode      = str(row.get("Payment_Mode", "")),
            vendor            = str(row.get("Vendor_Client_Name", "")),
        ))
    return facts


def run_rule_engine(df: pd.DataFrame,
                    category_col: str = "Category") -> FinancialSummary:
    """Convenience wrapper: DataFrame → rule engine → FinancialSummary."""
    facts = df_to_facts(df, category_col=category_col)
    engine = BackwardChainingRuleEngine(facts)
    return engine.run()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "financial_dataset_clean.csv"
    df = pd.read_csv(csv_path)
    summary = run_rule_engine(df)
    print("\nSummary dict:")
    for k, v in summary.to_dict().items():
        print(f"  {k}: {v}")
