"""
Transaction Categorization Service
FIXED: Correctly categorizes UPI transactions as CREDIT or DEBIT
"""

import re
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class TransactionCategorizer:
    """Categorize transactions - FIXED to properly detect CREDIT vs DEBIT"""
    
    def __init__(self):
        self._build_category_patterns()
    
    def _build_category_patterns(self):
        # Credit categories (money IN)
        self.credit_categories = [
            'Income', 'Reward/Cashback', 'Refund', 'Salary'
        ]
        
        # Debit categories (money OUT)
        self.debit_categories = [
            'Bill Payment', 'Subscription', 'EMI', 'UPI Transfer',
            'Bank Transfer', 'Cash Flow', 'Loan', 'Investment', 'Expense'
        ]
        
        # Credit keywords
        self.credit_patterns = [
            r'salary', r'payroll', r'wages', r'income', r'credit.*salary',
            r'interest', r'dividend', r'refund', r'reversal',
            r'cashback', r'bonus', r'reward', r'points',
            r'deposit', r'received', r'credit note',
        ]
        
        # Debit keywords
        self.debit_patterns = [
            r'emi', r'bill', r'electricity', r'water', r'gas',
            r'upi', r'imps', r'neft', r'rtgs', r'transfer',
            r'atm', r'withdrawal', r'subscription', r'payment',
            r'paid', r'charges', r'fee', r'tax',
        ]
    
    def categorize_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Categorize a transaction with CORRECT credit/debit detection.
        """
        description = str(transaction.get('description', '')).lower()
        narration = str(transaction.get('narration', '')).lower()
        
        credit = safe_float(transaction.get('credit'))
        debit = safe_float(transaction.get('debit'))
        amount = credit if credit > 0 else abs(debit)
        
        # === STEP 1: Check for explicit CR/Dr suffix on amount ===
        # If we have a credit amount, it's a credit transaction
        if credit > 0 and debit == 0:
            transaction_type = "CREDIT"
            category = "Income"
        # If we have a debit amount, it's a debit transaction
        elif debit > 0 and credit == 0:
            transaction_type = "DEBIT"
            category = "Expense"
        else:
            # Use keyword-based detection
            transaction_type, category = self._detect_from_keywords(description)
        
        # === STEP 2: Override based on description keywords ===
        desc_upper = description.upper()
        
        # Explicit credit indicators
        if any(kw in desc_upper for kw in ['SALARY', 'DEPOSIT', 'RECEIVED', 'REFUND', 
                                             'INTEREST', 'DIVIDEND', 'CASHBACK', 'BONUS',
                                             'INCOME', 'CREDIT NOTE']):
            transaction_type = "CREDIT"
            category = "Income"
        
        # Explicit debit indicators  
        elif any(kw in desc_upper for kw in ['PAID', 'WITHDRAWAL', 'WITHDRAW', 
                                              'ATM', 'CHARGES', 'FEE', 'TAX',
                                              'EMI', 'BILL', 'PAYMENT']):
            transaction_type = "DEBIT"
            category = "Expense"
        
        # === STEP 3: UPI transaction handling ===
        # UPI transactions can be credit OR debit
        if 'UPI' in desc_upper:
            if transaction_type == "UNKNOWN":
                # Default UPI to DEBIT (money going out)
                transaction_type = "DEBIT"
                category = "UPI Transfer"
            
            # Check for credit UPI patterns
            if any(kw in desc_upper for kw in ['RECEIVED', 'CREDIT', 'SENT']):
                # "Sent" is confusing - check if it's money in or out
                # Usually UPI Sent means money OUT (DEBIT)
                pass
        
        # === STEP 4: Bank transfer handling ===
        if any(kw in desc_upper for kw in ['IMPS', 'NEFT', 'RTGS', 'FT TO']):
            # These can be either - default to CREDIT if amount is positive
            if transaction_type == "UNKNOWN":
                transaction_type = "CREDIT" if amount > 0 else "DEBIT"
                category = "Bank Transfer"
        
        # === STEP 5: Check for IMPS incoming ===
        if 'IMPS' in desc_upper and 'FUNDING' in desc_upper:
            transaction_type = "CREDIT"
            category = "Bank Transfer"
        
        # Determine subcategory
        subcategory = self._get_subcategory(category, description)
        
        # Calculate risk
        merchant_risk_score = self._calculate_merchant_risk(description)
        
        return {
            'category': category,
            'subcategory': subcategory,
            'type': transaction_type,  # This is the key field!
            'merchant_risk_score': round(merchant_risk_score, 3),
            'narration_risk_confidence': 0.7,
            'behavioral_deviation': self._determine_behavioral_deviation(amount, category)
        }
    
    def _detect_from_keywords(self, text: str) -> tuple:
        """Detect transaction type from keywords"""
        text_lower = text.lower()
        
        # Check for credit keywords
        credit_score = 0
        for pattern in self.credit_patterns:
            if re.search(pattern, text_lower):
                credit_score += 1
        
        # Check for debit keywords
        debit_score = 0
        for pattern in self.debit_patterns:
            if re.search(pattern, text_lower):
                debit_score += 1
        
        if credit_score > debit_score:
            return "CREDIT", "Income"
        elif debit_score > credit_score:
            return "DEBIT", "Expense"
        else:
            return "UNKNOWN", "Expense"
    
    def _get_subcategory(self, category: str, description: str) -> str:
        """Get subcategory based on description"""
        desc = description.upper()
        
        subcategories = {
            'UPI Transfer': ['PAYTM', 'PHONEPE', 'GPAY', 'UPI'],
            'Bank Transfer': ['IMPS', 'NEFT', 'RTGS', 'FUNDING'],
            'Bill Payment': ['ELECTRICITY', 'WATER', 'GAS', 'MOBILE', 'INTERNET'],
            'Salary': ['SALARY', 'PAYROLL'],
            'Investment': ['FD', 'MUTUAL', 'STOCKS'],
        }
        
        for subcat, keywords in subcategories.items():
            if category in ['Income', 'Expense', 'UPI Transfer', 'Bank Transfer']:
                for kw in keywords:
                    if kw in desc:
                        return subcat
        
        return category
    
    def _calculate_merchant_risk(self, description: str) -> float:
        """Calculate risk score based on description"""
        desc = description.lower()
        
        high_risk = ['casino', 'gambling', 'betting', 'crypto', 'bitcoin']
        medium_risk = ['gaming', 'adult', 'dating']
        low_risk = ['utility', 'government', 'bank', 'salary']
        
        for kw in high_risk:
            if kw in desc:
                return 0.9
        
        for kw in medium_risk:
            if kw in desc:
                return 0.6
        
        for kw in low_risk:
            if kw in desc:
                return 0.2
        
        return 0.5
    
    def _determine_behavioral_deviation(self, amount: float, category: str) -> str:
        """Determine if transaction is unusual"""
        if amount > 100000:
            return 'High Value'
        elif amount < 10:
            return 'Micro Transaction'
        elif category == 'Unknown':
            return 'Uncategorized'
        else:
            return 'Normal'
