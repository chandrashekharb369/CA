"""
modules/rule_engine.py — CA Intelligence Suite
Phase 4: Backward Chaining Rule Engine — Schedule III (Companies Act 2013)

This module is the authoritative source for ALL financial computations.
It exposes:
    - BackwardChainingRuleEngine  — core inference engine
    - run_rule_engine(df)         — convenience wrapper
    - load_dynamic_rules(path)    — JSON-based compliance rule loader
    - evaluate_compliance_rule()  — per-transaction rule evaluator

Original file: rule_engine.py (root)
Modularised:   modules/rule_engine.py (this file)
Root file kept for backwards-compat with preprocess.py / train_model.py.
"""

# Re-export everything from the root rule_engine so existing imports continue
# to work (e.g. `from rule_engine import run_rule_engine`).
# New code should import from `modules.rule_engine`.

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from config import COMPLIANCE_RULES_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


# 
# Import core engine from root (single source of truth)
# 
try:
    # Import the full engine from the root module which contains the real impl
    from rule_engine import (   # noqa: F401  (root-level module)
        FinancialFact,
        FinancialSummary,
        BackwardChainingRuleEngine,
        S3_HEAD_FIELD_MAP,
        df_to_facts,
        run_rule_engine,
    )
    logger.debug("modules.rule_engine: core engine imported from root rule_engine.py")
except ImportError as exc:
    logger.error("modules.rule_engine: could not import root rule_engine — %s", exc)
    raise


# 
# Dynamic Compliance Rule System (Phase 4 upgrade)
# 

@dataclass
class ComplianceRule:
    """A single dynamic compliance rule loaded from JSON."""
    rule_name:  str
    condition:  str    # Python expression string — evaluated via eval()
    severity:   str    # "critical" | "warning" | "info"
    section:    str    # IT Act / CGST section reference
    message:    str    # Human-readable finding


def load_dynamic_rules(path: str = COMPLIANCE_RULES_PATH) -> List[ComplianceRule]:
    """
    Load compliance rules from a JSON file and return sorted by severity.

    Severity priority: critical > warning > info

    Args:
        path: Path to the JSON rules file.

    Returns:
        List of ComplianceRule objects, sorted by severity (critical first).

    Raises:
        FileNotFoundError: If the rules file does not exist.
    """
    if not os.path.exists(path):
        logger.warning("load_dynamic_rules: rules file not found at %s", path)
        return []

    try:
        with open(path, encoding="utf-8") as f:
            raw_rules = json.load(f)

        rules = [ComplianceRule(**r) for r in raw_rules]

        # Sort by severity priority: critical first, then warning, then info
        _priority = {"critical": 0, "warning": 1, "info": 2}
        rules.sort(key=lambda r: _priority.get(r.severity, 3))

        logger.info("load_dynamic_rules: loaded %d rules from %s", len(rules), path)
        return rules

    except Exception as exc:
        logger.error("load_dynamic_rules: failed to load rules — %s", exc, exc_info=True)
        return []


def evaluate_compliance_rule(transaction: pd.Series, rule: ComplianceRule) -> bool:
    """
    Evaluate a single compliance rule against one transaction row.

    Uses Python eval() with a sandboxed namespace containing only the
    transaction fields as local variables. This avoids arbitrary code execution
    while enabling flexible rule expressions.

    Safe namespace variables:
        amount       — transaction amount (float)
        payment_mode — payment mode string
        category     — transaction category
        sub_category — sub-category string
        description  — description string

    Args:
        transaction: A single row from a financial DataFrame (pd.Series).
        rule:        ComplianceRule to evaluate.

    Returns:
        True if the rule condition is satisfied, False otherwise.
    """
    # Build a restricted evaluation namespace
    safe_ns = {
        "amount":       float(transaction.get("Amount", 0)),
        "payment_mode": str(transaction.get("Payment_Mode", "")),
        "category":     str(transaction.get("Category", "")),
        "sub_category": str(transaction.get("Sub_Category", "")),
        "description":  str(transaction.get("Description", "")),
        # Allow basic builtins only
        "__builtins__": {},
    }

    try:
        return bool(eval(rule.condition, {"__builtins__": {}}, safe_ns))  # noqa: S307
    except Exception as exc:
        logger.debug(
            "evaluate_compliance_rule: rule '%s' eval error — %s",
            rule.rule_name, exc,
        )
        return False


def scan_transactions_for_violations(
    df: pd.DataFrame,
    rules: Optional[List[ComplianceRule]] = None,
) -> List[Dict]:
    """
    Scan all transactions against the dynamic rule set.

    Args:
        df:    Financial transactions DataFrame.
        rules: Optional pre-loaded rules list. Loads from JSON if None.

    Returns:
        List of violation dicts with keys:
            rule_name, severity, section, message, count, total_amount
    """
    if rules is None:
        rules = load_dynamic_rules()

    if not rules:
        return []

    violations: List[Dict] = []
    for rule in rules:
        try:
            # Apply rule to each row, collect matching rows
            mask   = df.apply(lambda row: evaluate_compliance_rule(row, rule), axis=1)
            hits   = df[mask]
            if not hits.empty:
                total_amt = hits["Amount"].sum() if "Amount" in hits.columns else 0.0
                violations.append({
                    "rule_name":    rule.rule_name,
                    "severity":     rule.severity,
                    "section":      rule.section,
                    "message":      rule.message,
                    "count":        int(len(hits)),
                    "total_amount": float(total_amt),
                })
        except Exception as exc:
            logger.warning("scan_transactions_for_violations: rule '%s' failed — %s",
                           rule.rule_name, exc)

    logger.info(
        "scan_transactions_for_violations: %d/%d rules triggered violations",
        len(violations), len(rules),
    )
    return violations
