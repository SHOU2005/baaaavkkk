"""
Universal Bank Statement PDF Analyzer
Enhanced Credit/Debit Detection - Returns LIST ONLY
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from io import BytesIO

import pdfplumber
import PyPDF2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_processor")


def safe_float(val, default=0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def extract_amounts_from_text(text: str) -> List[float]:
    """Extract all amounts from text"""
    amounts = []
    
    # Currency patterns
    currency_patterns = [r'[₹$€£¥]\s*([\d,]+\.?\d*)']
    
    for pattern in currency_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            cleaned = re.sub(r'[,\s]', '', match)
            try:
                amount = float(cleaned)
                if amount > 0:
                    amounts.append(amount)
            except ValueError:
                continue
    
    # Decimal patterns
    decimal_matches = re.findall(r'([\d,]+\.\d{2})', text)
    for match in decimal_matches:
        cleaned = re.sub(r'[,\s]', '', match)
        try:
            amount = float(cleaned)
            if 1 <= amount <= 999999:
                amounts.append(amount)
        except ValueError:
            continue
    
    return amounts


def detect_transaction_type(text: str) -> Tuple[str, float]:
    """
    Detect if transaction is CREDIT or DEBIT based on multiple signals.
    Returns (type, confidence)
    
    Priority order:
    1. Explicit CR/DR patterns (last occurrence wins)
    2. Keywords (SALARY, DEPOSIT, RECEIVED = CREDIT)
    3. Keywords (PAID, WITHDRAWAL, TRANSFER TO = DEBIT)
    4. Balance direction inference
    """
    text_upper = text.upper()
    
    credit_signals = 0.0
    debit_signals = 0.0
    reasons = []
    
    # ==== 1. CR/DR PATTERNS (highest priority) ====
    # Find ALL CR and DR occurrences and use the LAST one
    
    # CR patterns
    cr_patterns = [
        r'\bCR\b',           # CR as word
        r'\bCr\.\b',         # Cr. with period
        r'\bCREDIT\b',       # CREDIT keyword
        r'/CR/',             # /CR/ UPI pattern
    ]
    
    # DR patterns  
    dr_patterns = [
        r'\bDR\b',           # DR as word
        r'\bDr\.\b',         # Dr. with period
        r'\bDEBIT\b',        # DEBIT keyword
        r'/DR/',             # /DR/ UPI pattern
    ]
    
    cr_positions = []
    dr_positions = []
    
    for pattern in cr_patterns:
        for m in re.finditer(pattern, text_upper):
            cr_positions.append((m.start(), pattern))
    
    for pattern in dr_patterns:
        for m in re.finditer(pattern, text_upper):
            dr_positions.append((m.start(), pattern))
    
    last_cr = max([p[0] for p in cr_positions]) if cr_positions else -1
    last_dr = max([p[0] for p in dr_positions]) if dr_positions else -1
    
    if last_cr > last_dr:
        credit_signals += 3.0
        reasons.append(f"CR at position {last_cr}")
    elif last_dr > last_cr:
        debit_signals += 3.0
        reasons.append(f"DR at position {last_dr}")
    
    # ==== 2. CREDIT KEYWORDS ====
    credit_keywords = [
        (r'SALARY', 2.5),
        (r'SALARY\s+CR', 3.0),
        (r'INCOME', 2.0),
        (r'DEPOSIT', 2.0),
        (r'DEPOSITED', 2.0),
        (r'RECEIVED', 2.0),
        (r'REFUND', 2.5),
        (r'REVERSAL', 2.0),
        (r'CHARGEBACK', 2.5),
        (r'INTEREST', 2.0),
        (r'DIVIDEND', 2.0),
        (r'REWARD', 2.0),
        (r'CASHBACK', 2.5),
        (r'LOAN', 2.5),
        (r'CREDIT\s+NOTE', 2.5),
    ]
    
    for pattern, weight in credit_keywords:
        if re.search(pattern, text_upper):
            credit_signals += weight
            reasons.append(f"credit keyword: {pattern}")
    
    # ==== 3. DEBIT KEYWORDS ====
    debit_keywords = [
        (r'PAID', 2.5),
        (r'PAID\s+TO', 3.0),
        (r'PAYMENT', 2.0),
        (r'PAYMENT\s+TO', 2.5),
        (r'WITHDRAWAL', 2.5),
        (r'WITHDRAW', 2.5),
        (r'\sWDL\s', 2.5),
        (r'ATM', 1.5),
        (r'CASH\s+WITHDRAWAL', 3.0),
        (r'TRANSFER\s+TO', 2.5),
        (r'TO\s+[A-Z]', 1.5),
        (r'EMI', 2.5),
        (r'BILL', 2.0),
        (r'CHARGES', 2.0),
        (r'FEE', 2.0),
        (r'TAX', 2.0),
        (r'IMPS', 1.5),
        (r'NEFT', 1.5),
        (r'RTGS', 1.5),
        (r'UPI\s+[T|P]', 2.0),
    ]
    
    for pattern, weight in debit_keywords:
        if re.search(pattern, text_upper):
            debit_signals += weight
            reasons.append(f"debit keyword: {pattern}")
    
    # ==== 4. TRANSACTION CONTEXT ====
    # "TRANSFER TO" is debit, "TRANSFER FROM" is credit
    if re.search(r'TRANSFER\s+TO', text_upper):
        debit_signals += 2.0
        reasons.append("transfer to")
    elif re.search(r'TRANSFER\s+FROM', text_upper):
        credit_signals += 2.0
        reasons.append("transfer from")
    
    # ==== DETERMINE RESULT ====
    if credit_signals > debit_signals:
        return "CREDIT", credit_signals
    elif debit_signals > credit_signals:
        return "DEBIT", debit_signals
    else:
        # Tie - default to DEBIT but log warning
        return "DEBIT", 1.0


class PDFProcessor:
    """PDF processor - returns LIST ONLY with accurate credit/debit detection"""
    
    DATE_REGEX = re.compile(
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}-[A-Za-z]{3}-\d{2,4})\b'
    )
    
    AMOUNT_REGEX = re.compile(r'[\d,]+\.\d{2}')
    
    SKIP_WORDS = {
        "UPI", "IMPS", "NEFT", "RTGS", "DR", "CR", "DEBIT", "CREDIT",
        "TRANSFER", "PAYMENT", "WITHDRAWAL", "ATM", "WDL",
        "BANK", "INDIA", "ONLINE", "MOBILE"
    }
    
    def extract_transactions(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract transactions from PDF. Returns LIST ONLY."""
        text = self._extract_text(pdf_bytes)
        
        if not text.strip():
            return []
        
        return self._parse_transactions(text)
    
    def _extract_text(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF"""
        text = ""
        
        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
        
        if text.strip():
            return text
        
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        except Exception as e:
            logger.warning(f"PyPDF2 failed: {e}")
        
        return text
    
    def _parse_transactions(self, text: str) -> List[Dict[str, Any]]:
        """Parse transactions from extracted text"""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        transactions = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            date_match = self.DATE_REGEX.search(line)
            if not date_match:
                i += 1
                continue
            
            date = self._normalize_date(date_match.group(1))
            
            block = [line]
            j = i + 1
            while j < len(lines) and not self.DATE_REGEX.search(lines[j]):
                block.append(lines[j])
                j += 1
            
            txn = self._parse_block(date, block)
            if txn:
                transactions.append(txn)
            
            i = j
        
        return transactions
    
    def _parse_block(self, date: str, block: List[str]) -> Dict[str, Any]:
        """Parse a transaction block"""
        text = " ".join(block)
        
        # Extract amounts
        amounts = extract_amounts_from_text(text)
        
        if len(amounts) == 0:
            legacy_amounts = [float(a.replace(",", "")) for a in self.AMOUNT_REGEX.findall(text)]
            if len(legacy_amounts) > 0:
                amounts = [a for a in legacy_amounts if 1 <= a <= 999999]
        
        # Detect transaction type
        txn_type, confidence = detect_transaction_type(text)
        
        # Assign amounts based on type
        debit = 0.0
        credit = 0.0
        balance = 0.0
        
        if len(amounts) >= 1:
            if txn_type == "CREDIT":
                credit = amounts[0]
            else:  # DEBIT
                debit = amounts[0]
        
        if len(amounts) >= 2:
            balance = amounts[1]
        elif len(amounts) >= 3:
            balance = amounts[-1]
        
        # Calculate net amount
        if credit > 0:
            amount = credit
        elif debit > 0:
            amount = -debit
        else:
            amount = 0
        
        party = self._extract_party(text)
        description = self._clean_description(text)
        
        return {
            "date": date,
            "description": description,
            "party": party,
            "detected_party": party,
            "type": txn_type,
            "debit": debit,
            "credit": credit,
            "balance": balance,
            "amount": amount,
            "source": "pdf",
            "detection_confidence": confidence
        }
    
    def _extract_party(self, text: str) -> str:
        """Extract party name from transaction"""
        text = text.upper()
        
        patterns = [
            r'UPI/(?:CR|DR)/\d+/([A-Z\s]+)',
            r'IMPS/\d+/([A-Z\s]+)',
            r'NEFT/([A-Z\s]+)',
            r'TRANSFER\s+TO\s+([A-Z\s]+)',
            r'FROM\s+([A-Z\s]+)',
            r'PAID\s+TO\s+([A-Z\s]+)',
            r'RECEIVED\s+FROM\s+([A-Z\s]+)',
        ]
        
        for p in patterns:
            m = re.search(p, text)
            if m:
                return self._clean_party(m.group(1))
        
        words = [
            w for w in text.split()
            if w.isalpha() and len(w) > 3 and w not in self.SKIP_WORDS
        ]
        
        return " ".join(words[:4]) if words else "UNKNOWN"
    
    def _clean_party(self, name: str) -> str:
        """Clean party name"""
        name = re.sub(r'[^A-Z\s]', '', name)
        return " ".join(name.split()).strip()
    
    def _clean_description(self, text: str) -> str:
        """Clean transaction description"""
        text = self.DATE_REGEX.sub("", text)
        text = re.sub(r'[₹$€£¥]\s*[\d,]+\.?\d*', '', text)
        text = re.sub(r'[\d,]+\.\d{2}', '', text)
        return " ".join(text.split()).strip()
    
    def _normalize_date(self, raw: str) -> str:
        """Normalize date to DD/MM/YYYY"""
        raw = raw.replace("-", "/")
        d, m, y = raw.split("/")
        
        if len(y) == 2:
            y = "20" + y
        
        return f"{d.zfill(2)}/{m.zfill(2)}/{y}"
