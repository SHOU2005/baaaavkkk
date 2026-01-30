"""
Advanced Transaction Detection Module
Multi-signal detection with weighted scoring for accurate credit/debit classification
"""

import re
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class TransactionType(Enum):
    CREDIT = "credit"
    DEBIT = "debit"
    UNKNOWN = "unknown"


@dataclass
class DetectionSignal:
    """Represents a detection signal with its weight"""
    signal_type: str
    direction: TransactionType
    confidence: float
    pattern: str
    position: int


class AdvancedTransactionDetector:
    """
    Advanced transaction detector using multiple signals and weighted scoring.
    Designed to handle various bank statement formats (SBI, HDFC, ICICI, etc.)
    """
    
    def __init__(self):
        self._build_patterns()
    
    def _build_patterns(self):
        """Build all patterns for detection"""
        
        # Credit signals - patterns that indicate money IN
        self.credit_patterns = [
            # CR/DR patterns (high priority)
            (r'/CR/', 3.0, "UPI CR pattern"),
            (r'/CR\b', 2.5, "CR at word end"),
            (r'\bCR\b', 2.5, "CR word boundary"),
            (r'\bCr\.', 2.5, "Cr. with period"),
            (r'\bCREDIT\b', 2.5, "CREDIT keyword"),
            
            # Banking keywords
            (r'SALARY', 3.0, "Salary income"),
            (r'SALARY\s+CR', 3.5, "Salary CR"),
            (r'INCOME', 2.5, "Income"),
            (r'DEPOSIT', 2.0, "Deposit"),
            (r'DEPOSITED', 2.0, "Deposited"),
            (r'RECEIVED', 2.0, "Received"),
            (r'RECEIVE', 2.0, "Receive"),
            (r'REFUND', 2.5, "Refund"),
            (r'REVERSAL', 2.0, "Reversal"),
            (r'CHARGEBACK', 2.5, "Chargeback"),
            (r'CREDIT\s+NOTE', 2.5, "Credit note"),
            (r'INWARD', 2.0, "Inward transfer"),
            (r'FROM\s+[A-Z]', 1.5, "From party"),
            
            # Reward patterns
            (r'REWARD', 2.0, "Reward"),
            (r'CASHBACK', 2.5, "Cashback"),
            (r'BONUS', 2.0, "Bonus"),
            (r'LOYALTY', 1.5, "Loyalty"),
            (r'POINTS', 1.5, "Points"),
            
            # Loan related (money in)
            (r'LOAN\s+DISBURSEMENT', 3.0, "Loan disbursement"),
            (r'LOAN\s+CREDIT', 3.0, "Loan credit"),
        ]
        
        # Debit signals - patterns that indicate money OUT
        self.debit_patterns = [
            # CR/DR patterns (high priority)
            (r'/DR/', 3.0, "UPI DR pattern"),
            (r'/DR\b', 2.5, "DR at word end"),
            (r'\bDR\b', 2.5, "DR word boundary"),
            (r'\bDr\.', 2.5, "Dr. with period"),
            (r'\bDEBIT\b', 2.5, "DEBIT keyword"),
            
            # Banking keywords
            (r'PAID', 2.0, "Paid"),
            (r'PAYMENT', 1.5, "Payment"),
            (r'PAY\s+TO', 2.5, "Pay to"),
            (r'TO\s+[A-Z]', 1.5, "To party"),
            (r'WITHDRAWAL', 2.5, "Withdrawal"),
            (r'WITHDRAW', 2.5, "Withdraw"),
            (r'\sWDL\s', 2.5, "WDL pattern"),
            (r'ATM', 1.5, "ATM"),
            (r'CASH\s+WITHDRAWAL', 3.0, "Cash withdrawal"),
            (r'OUTWARD', 2.0, "Outward transfer"),
            
            # Bill/EMI patterns
            (r'EMI', 2.5, "EMI"),
            (r'BILL', 2.0, "Bill"),
            (r'ELECTRICITY', 2.0, "Electricity"),
            (r'WATER', 2.0, "Water"),
            (r'GAS', 2.0, "Gas"),
            (r'UTILITY', 2.0, "Utility"),
            
            # Transfer patterns
            (r'TRANSFER', 1.5, "Transfer"),
            (r'NEFT', 2.0, "NEFT"),
            (r'RTGS', 2.0, "RTGS"),
            (r'IMPS', 2.0, "IMPS"),
            (r'UPI\s+[T|P]', 2.0, "UPI payment"),
            
            # Subscription patterns
            (r'SUBSCRIPTION', 2.0, "Subscription"),
            (r'NETFLIX', 1.5, "Netflix"),
            (r'SPOTIFY', 1.5, "Spotify"),
            (r'AMAZON\s+PRIME', 1.5, "Amazon Prime"),
        ]
    
    def detect_transaction_type(
        self, 
        description: str, 
        credit_amount: float = 0.0,
        debit_amount: float = 0.0,
        amount: float = 0.0
    ) -> Tuple[TransactionType, float, List[DetectionSignal]]:
        """
        Detect transaction type using multiple signals with weighted scoring.
        
        Returns:
            Tuple of (transaction_type, confidence_score, signals)
        """
        signals: List[DetectionSignal] = []
        
        if not description:
            description = ""
        
        desc_upper = description.upper()
        
        # Signal 1: Column-based (most reliable for structured data)
        if credit_amount > 0 and debit_amount == 0:
            signals.append(DetectionSignal(
                signal_type="column",
                direction=TransactionType.CREDIT,
                confidence=1.0,
                pattern=f"credit_col={credit_amount}",
                position=0
            ))
        elif debit_amount > 0 and credit_amount == 0:
            signals.append(DetectionSignal(
                signal_type="column",
                direction=TransactionType.DEBIT,
                confidence=1.0,
                pattern=f"debit_col={debit_amount}",
                position=0
            ))
        
        # Signal 2: Sign-based (if negative amounts indicate debit)
        if amount < 0:
            signals.append(DetectionSignal(
                signal_type="sign",
                direction=TransactionType.DEBIT,
                confidence=0.9,
                pattern="negative_amount",
                position=0
            ))
        elif amount > 0:
            signals.append(DetectionSignal(
                signal_type="sign",
                direction=TransactionType.CREDIT,
                confidence=0.9,
                pattern="positive_amount",
                position=0
            ))
        
        # Signal 3: CR/DR pattern detection with position awareness
        self._detect_cr_dr_patterns(desc_upper, signals)
        
        # Signal 4: Keyword-based detection
        self._detect_keywords(desc_upper, signals)
        
        # Signal 5: Context patterns
        self._detect_context_patterns(desc_upper, signals)
        
        # Calculate weighted score
        credit_score = 0.0
        debit_score = 0.0
        
        for signal in signals:
            if signal.direction == TransactionType.CREDIT:
                credit_score += signal.confidence
            elif signal.direction == TransactionType.DEBIT:
                debit_score += signal.confidence
        
        # Determine winner
        if credit_score > debit_score:
            return TransactionType.CREDIT, credit_score, signals
        elif debit_score > credit_score:
            return TransactionType.DEBIT, debit_score, signals
        else:
            # Tie-breaker: use column data if available, else unknown
            for signal in signals:
                if signal.signal_type == "column":
                    return signal.direction, signal.confidence, signals
            return TransactionType.UNKNOWN, 0.0, signals
    
    def _detect_cr_dr_patterns(self, desc_upper: str, signals: List[DetectionSignal]):
        """Detect CR/DR patterns with position awareness - LAST occurrence wins"""
        
        # Find all CR and DR occurrences with their positions
        cr_positions = []
        dr_positions = []
        
        # CR patterns
        cr_patterns = [
            (r'/CR/', 3.0),
            (r'/CR\b', 2.5),
            (r'\bCR\b', 2.5),
            (r'\bCr\.', 2.5),
            (r'\bCREDIT\b', 2.5),
        ]
        
        # DR patterns
        dr_patterns = [
            (r'/DR/', 3.0),
            (r'/DR\b', 2.5),
            (r'\bDR\b', 2.5),
            (r'\bDr\.', 2.5),
            (r'\bDEBIT\b', 2.5),
        ]
        
        for pattern, weight in cr_patterns:
            for match in re.finditer(pattern, desc_upper):
                cr_positions.append((match.start(), weight, pattern))
        
        for pattern, weight in dr_patterns:
            for match in re.finditer(pattern, desc_upper):
                dr_positions.append((match.start(), weight, pattern))
        
        # CRITICAL FIX: Use POSITION-BASED priority, not just score-based
        # Find the LAST occurrence of BOTH CR and DR patterns
        last_cr_pos = max([p[0] for p in cr_positions]) if cr_positions else -1
        last_dr_pos = max([p[0] for p in dr_positions]) if dr_positions else -1
        
        # The one with the higher position (appears LATER) wins
        if last_cr_pos > last_dr_pos:
            # CR appears after DR, so this is a credit
            for pos, weight, pattern in cr_positions:
                if pos == last_cr_pos:
                    signals.append(DetectionSignal(
                        signal_type="cr_dr_pattern",
                        direction=TransactionType.CREDIT,
                        confidence=2.0,  # High confidence for position-based
                        pattern=f"CR (last): {pattern}",
                        position=pos
                    ))
                    break
        elif last_dr_pos > last_cr_pos:
            # DR appears after CR, so this is a debit
            for pos, weight, pattern in dr_positions:
                if pos == last_dr_pos:
                    signals.append(DetectionSignal(
                        signal_type="cr_dr_pattern",
                        direction=TransactionType.DEBIT,
                        confidence=2.0,  # High confidence for position-based
                        pattern=f"DR (last): {pattern}",
                        position=pos
                    ))
                    break
        else:
            # Only one type exists (or neither), add them as weak signals
            for pos, weight, pattern in cr_positions:
                signals.append(DetectionSignal(
                    signal_type="cr_dr_pattern",
                    direction=TransactionType.CREDIT,
                    confidence=0.5,  # Low confidence when no conflict
                    pattern=f"CR: {pattern}",
                    position=pos
                ))
            
            for pos, weight, pattern in dr_positions:
                signals.append(DetectionSignal(
                    signal_type="cr_dr_pattern",
                    direction=TransactionType.DEBIT,
                    confidence=0.5,  # Low confidence when no conflict
                    pattern=f"DR: {pattern}",
                    position=pos
                ))
    
    def _detect_keywords(self, desc_upper: str, signals: List[DetectionSignal]):
        """Detect transaction type from keywords"""
        
        # Credit keywords
        for pattern, weight, name in self.credit_patterns:
            if re.search(pattern, desc_upper):
                signals.append(DetectionSignal(
                    signal_type="keyword",
                    direction=TransactionType.CREDIT,
                    confidence=weight / 3.0,  # Normalize to 0-1
                    pattern=name,
                    position=0
                ))
        
        # Debit keywords
        for pattern, weight, name in self.debit_patterns:
            if re.search(pattern, desc_upper):
                signals.append(DetectionSignal(
                    signal_type="keyword",
                    direction=TransactionType.DEBIT,
                    confidence=weight / 3.0,  # Normalize to 0-1
                    pattern=name,
                    position=0
                ))
    
    def _detect_context_patterns(self, desc_upper: str, signals: List[DetectionSignal]):
        """Detect from contextual patterns"""
        
        # TO/FROM patterns
        from_match = re.search(r'FROM\s+([A-Z][A-Z\s]{2,})', desc_upper)
        if from_match:
            signals.append(DetectionSignal(
                signal_type="context",
                direction=TransactionType.CREDIT,
                confidence=0.7,
                pattern="FROM pattern",
                position=from_match.start()
            ))
        
        to_match = re.search(r'TO\s+([A-Z][A-Z\s]{2,})', desc_upper)
        if to_match:
            signals.append(DetectionSignal(
                signal_type="context",
                direction=TransactionType.DEBIT,
                confidence=0.7,
                pattern="TO pattern",
                position=to_match.start()
            ))
        
        # UPI reference number patterns (should be low confidence)
        upi_cr_match = re.search(r'UPI/\w+/CR/\d+', desc_upper)
        if upi_cr_match:
            signals.append(DetectionSignal(
                signal_type="upi_ref",
                direction=TransactionType.CREDIT,
                confidence=0.6,
                pattern="UPI CR ref",
                position=0
            ))
        
        upi_dr_match = re.search(r'UPI/\w+/DR/\d+', desc_upper)
        if upi_dr_match:
            signals.append(DetectionSignal(
                signal_type="upi_ref",
                direction=TransactionType.DEBIT,
                confidence=0.6,
                pattern="UPI DR ref",
                position=0
            ))
        
        # Balance indicator patterns
        if re.search(r'\d+\.\d{2}\s+DR$', desc_upper):
            signals.append(DetectionSignal(
                signal_type="balance_indicator",
                direction=TransactionType.DEBIT,
                confidence=0.8,
                pattern="amount DR suffix",
                position=len(desc_upper)
            ))
        
        if re.search(r'\d+\.\d{2}\s+CR$', desc_upper):
            signals.append(DetectionSignal(
                signal_type="balance_indicator",
                direction=TransactionType.CREDIT,
                confidence=0.8,
                pattern="amount CR suffix",
                position=len(desc_upper)
            ))


# Singleton instance for reuse
_detector_instance: Optional[AdvancedTransactionDetector] = None


def get_detector() -> AdvancedTransactionDetector:
    """Get the singleton detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AdvancedTransactionDetector()
    return _detector_instance


def detect_transaction_type(
    description: str,
    credit_amount: float = 0.0,
    debit_amount: float = 0.0,
    amount: float = 0.0
) -> Tuple[TransactionType, float]:
    """
    Convenience function for transaction type detection.
    
    Args:
        description: Transaction description/narration
        credit_amount: Credit column value (if available)
        debit_amount: Debit column value (if available)
        amount: Amount value (if available)
    
    Returns:
        Tuple of (transaction_type, confidence)
    """
    detector = get_detector()
    txn_type, confidence, _ = detector.detect_transaction_type(
        description, credit_amount, debit_amount, amount
    )
    return txn_type, confidence

