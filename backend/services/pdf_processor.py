"""
Universal Bank Statement PDF Analyzer
Returns LIST ONLY for compatibility
"""

import re
import logging
from typing import List, Dict, Any
from io import BytesIO

import pdfplumber
import PyPDF2
from services.transaction_detector import AdvancedTransactionDetector, TransactionType, get_detector

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


def extract_amount_from_text(text: str) -> List[float]:
    """Extract all amounts from text"""
    amounts = []
    
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


def detect_transaction_direction(text: str) -> str:
    """Detect if a transaction is a credit or debit"""
    text_upper = text.upper()
    
    cr_patterns = [r'\bCR\b', r'\bCr\.\b', r'\bCREDIT\b', r'/CR/']
    dr_patterns = [r'\bDR\b', r'\bDr\.\b', r'\bDEBIT\b', r'/DR/']
    
    all_cr_positions = []
    all_dr_positions = []
    
    for pattern in cr_patterns:
        for m in re.finditer(pattern, text_upper):
            all_cr_positions.append(m.start())
    
    for pattern in dr_patterns:
        for m in re.finditer(pattern, text_upper):
            all_dr_positions.append(m.start())
    
    last_cr = max(all_cr_positions) if all_cr_positions else -1
    last_dr = max(all_dr_positions) if all_dr_positions else -1
    
    if last_cr > last_dr:
        return 'credit'
    elif last_dr > last_cr:
        return 'debit'
    else:
        if 'PAID TO' in text_upper:
            return 'debit'
        elif 'RECEIVED FROM' in text_upper:
            return 'credit'
        elif 'DEPOSIT' in text_upper:
            return 'credit'
        elif 'WITHDRAWAL' in text_upper or ' WDL ' in text_upper:
            return 'debit'
        return 'unknown'


class PDFProcessor:
    """PDF processor - returns LIST ONLY"""
    
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
        """
        Extract transactions from PDF. Returns LIST ONLY.
        """
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
        
        amounts = extract_amount_from_text(text)
        
        debit = credit = balance = 0.0
        
        if len(amounts) == 0:
            legacy_amounts = [float(a.replace(",", "")) for a in self.AMOUNT_REGEX.findall(text)]
            if len(legacy_amounts) > 0:
                amounts = [a for a in legacy_amounts if 1 <= a <= 999999]
        
        detector = get_detector()
        # detect_transaction_type returns (TransactionType, confidence, signals) - 3 values!
        result = detector.detect_transaction_type(
            text, 0.0, 0.0, amounts[0] if amounts else 0.0
        )
        txn_type = result[0]  # Get only the transaction type
        
        if len(amounts) == 1:
            if txn_type == TransactionType.CREDIT:
                credit = amounts[0]
            elif txn_type == TransactionType.DEBIT:
                debit = amounts[0]
            else:
                debit = amounts[0]
        elif len(amounts) == 2:
            if txn_type == TransactionType.CREDIT:
                credit = amounts[0]
            elif txn_type == TransactionType.DEBIT:
                debit = amounts[0]
            else:
                debit = amounts[0]
            balance = amounts[1]
        elif len(amounts) >= 3:
            if txn_type == TransactionType.CREDIT:
                credit = amounts[0]
            elif txn_type == TransactionType.DEBIT:
                debit = amounts[0]
            else:
                debit = amounts[0]
            balance = amounts[-1]
        
        if credit > 0:
            amount = credit
        elif debit > 0:
            amount = -debit
        else:
            amount = 0
        
        txn_type_str = self._detect_type(text, debit, credit, txn_type)
        party = self._extract_party(text)
        description = self._clean_description(text)
        
        return {
            "date": date,
            "description": description,
            "party": party,
            "detected_party": party,
            "type": txn_type_str,
            "debit": debit if debit > 0 else 0.0,
            "credit": credit if credit > 0 else 0.0,
            "balance": balance,
            "amount": amount,
            "source": "pdf"
        }
    
    def _detect_type(self, text: str, debit, credit, detected_type=None) -> str:
        """Detect transaction type string"""
        t = text.upper()
        
        if detected_type is not None:
            if detected_type == TransactionType.CREDIT:
                return "CREDIT"
            elif detected_type == TransactionType.DEBIT:
                return "DEBIT"
        
        if any(x in t for x in [" CREDIT", " CR", " DEPOSIT", "SALARY", "INCOME"]):
            return "CREDIT"
        if any(x in t for x in [" DEBIT", " DR", " WDL", " WITHDRAWAL", "PAID"]):
            return "DEBIT"
        
        if credit and not debit:
            return "CREDIT"
        return "DEBIT"
    
    def _extract_party(self, text: str) -> str:
        """Extract party name from transaction"""
        text = text.upper()
        
        patterns = [
            r'UPI/(?:CR|DR)/\d+/([A-Z\s]+)',
            r'IMPS/\d+/([A-Z\s]+)',
            r'NEFT/([A-Z\s]+)',
            r'TRANSFER TO ([A-Z\s]+)',
            r'FROM ([A-Z\s]+)'
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
