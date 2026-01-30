"""
Transaction Validation Engine
Validates transactions for accuracy and consistency
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class ValidationReport:
    """Summary of validation results"""
    total_transactions: int
    valid_transactions: int
    invalid_transactions: int
    credit_debit_mismatches: int
    missing_amounts: int
    missing_dates: int
    missing_parties: int
    warnings: List[str]
    suggestions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_transactions": self.total_transactions,
            "valid_transactions": self.valid_transactions,
            "invalid_transactions": self.invalid_transactions,
            "credit_debit_mismatches": self.credit_debit_mismatches,
            "missing_amounts": self.missing_amounts,
            "missing_dates": self.missing_dates,
            "missing_parties": self.missing_parties,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "is_valid": self.invalid_transactions == 0
        }


class ValidationEngine:
    """Validates transactions for common errors"""
    
    def __init__(self):
        self.credit_keywords = ['SALARY', 'DEPOSIT', 'RECEIVED', 'REFUND', 
                                'INTEREST', 'DIVIDEND', 'INCOME', 'CREDIT']
        self.debit_keywords = ['PAID', 'WITHDRAWAL', 'WITHDRAW', 'ATM',
                               'CHARGES', 'FEE', 'TAX', 'EMI', 'BILL']
    
    def validate_transactions(self, transactions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], ValidationReport]:
        """
        Validate all transactions and return cleaned list with report.
        
        Returns:
            Tuple of (validated_transactions, validation_report)
        """
        valid = []
        report = ValidationReport(
            total_transactions=len(transactions),
            valid_transactions=0,
            invalid_transactions=0,
            credit_debit_mismatches=0,
            missing_amounts=0,
            missing_dates=0,
            missing_parties=0,
            warnings=[],
            suggestions=[]
        )
        
        for idx, txn in enumerate(transactions):
            try:
                # Make a copy to avoid modifying original
                validated = txn.copy()
                
                # Validate and fix credit/debit
                is_valid, mismatch = self._validate_credit_debit(validated)
                if mismatch:
                    report.credit_debit_mismatches += 1
                
                # Check for missing amounts
                if self._has_missing_amount(validated):
                    report.missing_amounts += 1
                    self._fix_missing_amount(validated)
                
                # Check for missing dates
                if not validated.get('date'):
                    report.missing_dates += 1
                
                # Check for missing parties
                if not validated.get('party') or validated.get('party') == 'UNKNOWN':
                    report.missing_parties += 1
                    self._fix_missing_party(validated)
                
                # Ensure type is valid
                if validated.get('type') not in ['CREDIT', 'DEBIT', 'UNKNOWN']:
                    validated['type'] = 'UNKNOWN'
                
                # Mark as valid
                validated['_validation_passed'] = True
                valid.append(validated)
                report.valid_transactions += 1
                
            except Exception as e:
                logger.warning(f"Validation error at index {idx}: {str(e)}")
                report.invalid_transactions += 1
                # Still include the transaction but mark as invalid
                txn['_validation_passed'] = False
                txn['_validation_error'] = str(e)
                valid.append(txn)
        
        # Generate warnings and suggestions
        self._generate_warnings(report, transactions)
        
        return valid, report
    
    def _validate_credit_debit(self, txn: Dict[str, Any]) -> Tuple[bool, bool]:
        """
        Validate credit/debit assignment.
        Returns (is_valid, has_mismatch)
        """
        credit = safe_float(txn.get('credit'))
        debit = safe_float(txn.get('debit'))
        amount = safe_float(txn.get('amount'))
        txn_type = txn.get('type', 'UNKNOWN')
        description = str(txn.get('description', '')).upper()
        
        has_mismatch = False
        
        # Check for contradictory values
        if credit > 0 and debit > 0:
            # Both have values - that's a mismatch
            has_mismatch = True
            # Use the larger one
            if credit >= debit:
                debit = 0
                txn['debit'] = 0
                txn['type'] = 'CREDIT'
            else:
                credit = 0
                txn['credit'] = 0
                txn['type'] = 'DEBIT'
        
        # Check type consistency with description
        if txn_type == 'CREDIT':
            # Should not have debit-only keywords
            for kw in self.debit_keywords:
                if kw in description and 'WITHDRAWAL' not in description:
                    # Could be valid (e.g., "ATM Withdrawal" could be debit)
                    pass
        
        if txn_type == 'DEBIT':
            # Should not have credit-only keywords
            for kw in self.credit_keywords:
                if kw in description:
                    # Likely a mismatch
                    has_mismatch = True
        
        return True, has_mismatch
    
    def _has_missing_amount(self, txn: Dict[str, Any]) -> bool:
        """Check if transaction has missing or zero amounts"""
        credit = safe_float(txn.get('credit'))
        debit = safe_float(txn.get('debit'))
        amount = safe_float(txn.get('amount'))
        return credit == 0 and debit == 0 and amount == 0
    
    def _fix_missing_amount(self, txn: Dict[str, Any]):
        """Try to fix missing amounts from description"""
        description = str(txn.get('description', ''))
        
        # Try to extract amount from description
        amount_patterns = [
            r'[₹$€£¥]\s*([\d,]+\.?\d*)',
            r'₹\s*([\d,]+\.?\d*)',
            r'RS\.?\s*([\d,]+\.?\d*)',
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                amount_str = re.sub(r'[,\s]', '', match.group(1))
                try:
                    amount = float(amount_str)
                    if amount > 0:
                        txn['amount'] = amount
                        # Try to determine type
                        if 'PAID' in description.upper() or 'DEBIT' in description.upper():
                            txn['debit'] = amount
                            txn['type'] = 'DEBIT'
                        elif 'CREDIT' in description.upper() or 'RECEIVED' in description.upper():
                            txn['credit'] = amount
                            txn['type'] = 'CREDIT'
                        break
                except ValueError:
                    continue
        
        if not txn.get('amount'):
            txn['amount'] = 0
    
    def _has_missing_date(self, txn: Dict[str, Any]) -> bool:
        """Check if date is missing"""
        return not txn.get('date') or txn.get('date') == ''
    
    def _fix_missing_date(self, txn: Dict[str, Any]):
        """Try to fix missing dates"""
        # Can't fix if no date info available
        if not txn.get('date'):
            txn['date'] = '01/01/1900'
    
    def _has_missing_party(self, txn: Dict[str, Any]) -> bool:
        """Check if party is missing"""
        party = txn.get('party', '')
        return not party or party == '' or party == 'UNKNOWN'
    
    def _fix_missing_party(self, txn: Dict[str, Any]):
        """Try to fix missing party from description"""
        description = str(txn.get('description', ''))
        
        # Try to extract party from common patterns
        patterns = [
            r'(?:TO|FROM|AT)\s+([A-Z][A-Z\s]{2,})',
            r'UPI/\w+/([A-Z]+)',
            r'@([A-Z0-9]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                party = match.group(1).strip()
                party = re.sub(r'[^\w\s]', '', party)
                party = ' '.join(party.split()).strip()
                if len(party) >= 2:
                    txn['party'] = party
                    return
        
        # Default
        txn['party'] = 'UNKNOWN'
    
    def _generate_warnings(self, report: ValidationReport, transactions: List[Dict[str, Any]]):
        """Generate warnings and suggestions based on validation results"""
        
        if report.missing_amounts > 0:
            report.warnings.append(f"{report.missing_amounts} transactions had missing amounts")
            report.suggestions.append("Review transactions with ₹0 amounts")
        
        if report.credit_debit_mismatches > 0:
            report.warnings.append(f"{report.credit_debit_mismatches} transactions had credit/debit conflicts")
            report.suggestions.append("Verify credit/debit assignments for conflicting transactions")
        
        if report.missing_dates > 0:
            report.warnings.append(f"{report.missing_dates} transactions had missing dates")
            report.suggestions.append("Add dates for transactions with missing date info")
        
        if report.missing_parties > 0:
            report.warnings.append(f"{report.missing_parties} transactions had missing parties")
            report.suggestions.append("Review party assignments for transactions with UNKNOWN party")
        
        # Overall assessment
        if report.invalid_transactions > report.total_transactions * 0.1:
            report.warnings.append("High error rate detected - consider manual review")
        
        # Positive feedback
        if report.valid_transactions == report.total_transactions:
            report.suggestions.append("All transactions passed validation")
        
        report.suggestions.append("Cross-check total credits and debits with statement totals")
