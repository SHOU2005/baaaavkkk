"""
Balance Tracker for Bank Statement Validation
Tracks running balance and validates transaction consistency.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BalanceCheckpoint:
    """A checkpoint in the balance tracking"""
    transaction_index: int
    date: str
    balance: float
    transaction_amount: float
    transaction_type: str
    is_valid: bool
    discrepancy: float = 0.0
    note: str = ""


@dataclass
class BalanceGap:
    """Represents a gap or inconsistency in balance"""
    start_index: int
    end_index: int
    expected_balance: float
    actual_balance: float
    discrepancy: float
    transactions_in_gap: List[int]
    possible_fixes: List[str]
    severity: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "expected_balance": round(self.expected_balance, 2),
            "actual_balance": round(self.actual_balance, 2),
            "discrepancy": round(self.discrepancy, 2),
            "transactions_in_gap": self.transactions_in_gap,
            "possible_fixes": self.possible_fixes,
            "severity": self.severity
        }


@dataclass
class StatementBalanceSummary:
    """Summary of balance analysis for a statement"""
    opening_balance: float
    closing_balance: float
    total_credits: float
    total_debits: float
    net_change: float
    expected_closing: float
    discrepancy: float
    is_balanced: bool
    gap_count: int
    invalid_balance_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "opening_balance": round(self.opening_balance, 2),
            "closing_balance": round(self.closing_balance, 2),
            "total_credits": round(self.total_credits, 2),
            "total_debits": round(self.total_debits, 2),
            "net_change": round(self.net_change, 2),
            "expected_closing": round(self.expected_closing, 2),
            "discrepancy": round(self.discrepancy, 2),
            "is_balanced": self.is_balanced,
            "gap_count": self.gap_count,
            "invalid_balance_count": self.invalid_balance_count
        }


class BalanceTracker:
    """Tracks running balance across transactions and validates consistency"""
    
    def __init__(self):
        self.checkpoints: List[BalanceCheckpoint] = []
        self.gaps: List[BalanceGap] = []
        self.balance_sequence: List[Dict[str, Any]] = []
        self.total_discrepancy = 0.0
        self.gap_count = 0
        self.valid_balance_count = 0
    
    def clear(self):
        self.checkpoints.clear()
        self.gaps.clear()
        self.balance_sequence.clear()
        self.total_discrepancy = 0.0
        self.gap_count = 0
        self.valid_balance_count = 0
    
    def _safe_float(self, value, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).replace(',', ''))
        except (ValueError, TypeError):
            return default
    
    def calculate_running_balance(
        self, 
        transactions: List[Dict[str, Any]], 
        initial_balance: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        self.clear()
        
        current_balance = initial_balance
        running_balances = []
        
        for idx, txn in enumerate(transactions):
            credit = self._safe_float(txn.get('credit', 0))
            debit = self._safe_float(txn.get('debit', 0))
            stated_balance = self._safe_float(txn.get('balance', None))
            
            if credit > 0:
                txn_type = 'credit'
                amount = credit
            elif debit > 0:
                txn_type = 'debit'
                amount = -debit
            else:
                txn_type = 'unknown'
                amount = self._safe_float(txn.get('amount', 0))
            
            if current_balance is not None:
                current_balance += amount
            
            is_valid = True
            discrepancy = 0.0
            note = ""
            
            if stated_balance is not None and stated_balance > 0:
                if current_balance is not None:
                    discrepancy = abs(stated_balance - current_balance)
                    if discrepancy > 1.0:
                        is_valid = False
                        self.total_discrepancy += discrepancy
                    else:
                        current_balance = stated_balance
                        note = "Using stated balance"
                else:
                    current_balance = stated_balance
                    note = "Initialized from statement"
            
            checkpoint = BalanceCheckpoint(
                transaction_index=idx,
                date=txn.get('date', ''),
                balance=current_balance or 0.0,
                transaction_amount=abs(amount),
                transaction_type=txn_type,
                is_valid=is_valid,
                discrepancy=discrepancy,
                note=note
            )
            self.checkpoints.append(checkpoint)
            
            if is_valid:
                self.valid_balance_count += 1
            
            txn_copy = txn.copy()
            txn_copy['running_balance'] = current_balance
            txn_copy['balance_valid'] = is_valid
            txn_copy['balance_discrepancy'] = discrepancy
            running_balances.append(txn_copy)
        
        self.balance_sequence = running_balances
        return running_balances
    
    def detect_balance_gaps(
        self, 
        transactions: List[Dict[str, Any]],
        tolerance: float = 1.0
    ) -> List[BalanceGap]:
        self.gaps.clear()
        
        gap_start = None
        gap_transactions = []
        
        for idx, txn in enumerate(transactions):
            stated_balance = self._safe_float(txn.get('balance', None))
            balance_valid = txn.get('balance_valid', True)
            
            if not balance_valid and stated_balance is not None:
                if gap_start is None:
                    gap_start = idx
                    gap_transactions = [idx]
                else:
                    gap_transactions.append(idx)
            else:
                if gap_start is not None and len(gap_transactions) > 0:
                    prev_txn = transactions[max(0, gap_start - 1)]
                    stated_next = self._safe_float(
                        transactions[min(len(transactions) - 1, gap_start)].get('balance', None)
                    )
                    prev_balance = self._safe_float(prev_txn.get('running_balance', 0))
                    
                    if stated_next is not None and prev_balance is not None:
                        expected_balance = prev_balance
                        
                        for t_idx in gap_transactions:
                            t_credit = self._safe_float(transactions[t_idx].get('credit', 0))
                            t_debit = self._safe_float(transactions[t_idx].get('debit', 0))
                            expected_balance += (t_credit - t_debit)
                        
                        discrepancy = abs(stated_next - expected_balance)
                        
                        gap = BalanceGap(
                            start_index=gap_start,
                            end_index=gap_transactions[-1],
                            expected_balance=expected_balance,
                            actual_balance=stated_next,
                            discrepancy=discrepancy,
                            transactions_in_gap=gap_transactions.copy(),
                            possible_fixes=[
                                f"Review transactions {gap_start + 1} to {gap_transactions[-1] + 1}",
                                "Check for missing amounts in gap transactions"
                            ],
                            severity='high' if discrepancy > 100 else 'medium' if discrepancy > 10 else 'low'
                        )
                        self.gaps.append(gap)
                
                gap_start = None
                gap_transactions = []
        
        self.gap_count = len(self.gaps)
        return self.gaps
    
    def get_balance_statistics(self) -> Dict[str, Any]:
        return {
            'total_checkpoints': len(self.checkpoints),
            'valid_balances': self.valid_balance_count,
            'invalid_balances': len(self.checkpoints) - self.valid_balance_count,
            'total_discrepancy': round(self.total_discrepancy, 2),
            'gap_count': self.gap_count,
            'validity_rate': round(
                self.valid_balance_count / max(len(self.checkpoints), 1) * 100, 1
            )
        }


def analyze_statement_balance(
    transactions: List[Dict[str, Any]],
    opening_balance: Optional[float] = None,
    closing_balance: Optional[float] = None
) -> StatementBalanceSummary:
    """Analyze complete statement balance"""
    tracker = BalanceTracker()
    
    total_credits = sum(
        tracker._safe_float(t.get('credit', 0)) for t in transactions
    )
    total_debits = sum(
        tracker._safe_float(t.get('debit', 0)) for t in transactions
    )
    
    if opening_balance is not None:
        expected_closing = opening_balance + total_credits - total_debits
    else:
        expected_closing = total_credits - total_debits
    
    discrepancy = 0.0
    if closing_balance is not None:
        discrepancy = abs(closing_balance - expected_closing)
    
    is_balanced = discrepancy <= 1.0
    
    return StatementBalanceSummary(
        opening_balance=opening_balance or 0.0,
        closing_balance=closing_balance or expected_closing,
        total_credits=total_credits,
        total_debits=total_debits,
        net_change=total_credits - total_debits,
        expected_closing=expected_closing,
        discrepancy=discrepancy,
        is_balanced=is_balanced,
        gap_count=tracker.gap_count,
        invalid_balance_count=len(transactions) - tracker.valid_balance_count
    )
