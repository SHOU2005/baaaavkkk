"""
Universal Bank Statement PDF Analyzer
Fixed Credit/Debit Detection - Prioritizes Amount Suffix
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


def extract_amounts_from_text(text: str) -> List[Tuple[float, str]]:
    """
    Extract amounts with their suffix (Cr/Dr) from text.
    Returns list of (amount, suffix) tuples.
    """
    results = []
    
    # Pattern: amount followed by Cr or Dr (with optional spaces)
    # Matches: "9605.00 Cr", "26605.00Cr", "500.00 Cr", etc.
    amount_with_suffix = re.findall(r'([\d,]+\.\d{2})\s*(Cr|Dr|CR|DR|cr|dr)\b', text)
    
    for amount_str, suffix in amount_with_suffix:
        cleaned = re.sub(r'[,\s]', '', amount_str)
        try:
            amount = float(cleaned)
            if amount > 0:
                results.append((amount, suffix.upper()))
        except ValueError:
            continue
    
    # Also extract amounts without suffix for fallback
    decimal_matches = re.findall(r'([\d,]+\.\d{2})(?!\s*(?:Cr|Dr|CR|DR))', text)
    for match in decimal_matches:
        cleaned = re.sub(r'[,\s]', '', match)
        try:
            amount = float(cleaned)
            if 1 <= amount <= 999999:
                # Check if this amount is followed by "Cr" or "Dr" nearby
                pos = text.find(match)
                if pos > 0:
                    nearby = text[pos:pos+20]
                    if 'Cr' in nearby or 'CR' in nearby:
                        results.append((amount, 'CR'))
                    elif 'Dr' in nearby or 'DR' in nearby:
                        results.append((amount, 'DR'))
                    else:
                        results.append((amount, ''))
        except ValueError:
            continue
    
    return results


def detect_transaction_type(text: str) -> Tuple[str, float, List[str]]:
    """
    Detect if transaction is CREDIT or DEBIT.
    
    Priority:
    1. Amount suffix (Cr/Dr) - MOST RELIABLE
    2. CR/DR in narration (position-based)
    3. Keywords
    """
    text_upper = text.upper()
    reasons = []
    
    # ==== 1. CHECK AMOUNT SUFFIX FIRST (most reliable) ====
    amounts_with_suffix = extract_amounts_from_text(text)
    
    # Look for amounts with Cr or Dr suffix
    credit_amounts = []
    debit_amounts = []
    unknown_amounts = []
    
    for amount, suffix in amounts_with_suffix:
        if suffix == 'CR':
            credit_amounts.append(amount)
        elif suffix == 'DR':
            debit_amounts.append(amount)
        else:
            unknown_amounts.append(amount)
    
    # If we found credit amounts, this is a credit transaction
    if credit_amounts:
        txn_type = "CREDIT"
        confidence = 3.0
        reasons.append(f"amount suffix Cr: {credit_amounts[0]}")
        return txn_type, confidence, reasons
    
    # If we found debit amounts, this is a debit transaction
    if debit_amounts:
        txn_type = "DEBIT"
        confidence = 3.0
        reasons.append(f"amount suffix Dr: {debit_amounts[0]}")
        return txn_type, confidence, reasons
    
    # ==== 2. CHECK CR/DR PATTERNS (last occurrence wins) ====
    cr_patterns = [r'\bCR\b', r'/CR/', r'\bCr\.\b']
    dr_patterns = [r'\bDR\b', r'/DR/', r'\bDr\.\b']
    
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
        reasons.append(f"CR at position {last_cr}")
        # Verify this isn't just a UPI reference number
        # UPI references often look like: UPI/DR/123456 - the DR here is reference, not type
        # If CR comes after DR, use CR. If CR comes before DR, we need to look closer
        return "CREDIT", 2.5, reasons
    elif last_dr > last_cr:
        reasons.append(f"DR at position {last_dr}")
        return "DEBIT", 2.5, reasons
    
    # ==== 3. CHECK KEYWORDS ====
    credit_keywords = [
        (r'SALARY', 2.0), (r'DEPOSIT', 2.0), (r'RECEIVED', 2.0),
        (r'REFUND', 2.0), (r'INTEREST', 2.0), (r'DIVIDEND', 2.0),
        (r'CASHBACK', 2.0), (r'BONUS', 2.0), (r'LOAN', 2.0),
    ]
    
    debit_keywords = [
        (r'PAID', 2.0), (r'PAID\s+TO', 2.5), (r'WITHDRAWAL', 2.0),
        (r'\sWDL\s', 2.0), (r'TRANSFER\s+TO', 2.0), (r'TO\s+[A-Z]', 1.5),
        (r'EMI', 2.0), (r'BILL', 1.5), (r'CHARGES', 1.5),
    ]
    
    credit_signals = 0.0
    debit_signals = 0.0
    
    for pattern, weight in credit_keywords:
        if re.search(pattern, text_upper):
            credit_signals += weight
            reasons.append(f"credit keyword: {pattern}")
    
    for pattern, weight in debit_keywords:
        if re.search(pattern, text_upper):
            debit_signals += weight
            reasons.append(f"debit keyword: {pattern}")
    
    if credit_signals > debit_signals:
        return "CREDIT", credit_signals, reasons
    elif debit_signals > credit_signals:
        return "DEBIT", debit_signals, reasons
    
    # ==== 4. DEFAULT ====
    # If no clear indicator, default to DEBIT
    reasons.append("default to DEBIT")
    return "DEBIT", 1.0, reasons


class PDFProcessor:
    """PDF processor - Accurate credit/debit detection"""
    
    DATE_REGEX = re.compile(
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}-[A-Za-z]{3}-\d{2,4})\b'
    )
    
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
        
        # Detect transaction type FIRST (this uses amount suffix)
        txn_type, confidence, detection_reasons = detect_transaction_type(text)
        
        # Extract amounts
        amounts_with_suffix = extract_amounts_from_text(text)
        
        # Separate credit and debit amounts
        credit_amounts = [a for a, s in amounts_with_suffix if s == 'CR']
        debit_amounts = [a for a, s in amounts_with_suffix if s == 'DR']
        other_amounts = [a for a, s in amounts_with_suffix if s == '']
        
        # Assign amounts based on detected type
        debit = 0.0
        credit = 0.0
        balance = 0.0
        
        if txn_type == "CREDIT":
            # Use credit amounts, or first other amount
            if credit_amounts:
                credit = credit_amounts[0]
            elif other_amounts:
                credit = other_amounts[0]
            
            # Balance is typically the last amount
            all_amounts = [a for a, _ in amounts_with_suffix]
            if len(all_amounts) >= 2:
                balance = all_amounts[-1]
                
        else:  # DEBIT
            # Use debit amounts, or first other amount
            if debit_amounts:
                debit = debit_amounts[0]
            elif other_amounts:
                debit = other_amounts[0]
            
            # Balance is typically the last amount
            all_amounts = [a for a, _ in amounts_with_suffix]
            if len(all_amounts) >= 2:
                balance = all_amounts[-1]
        
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
            "detection_confidence": confidence,
            "detection_reasons": detection_reasons
        }
    
    def _extract_party(self, text: str) -> str:
        """Extract party name from transaction"""
        text = text.upper()
        
        patterns = [
            r'UPI/\w+/([A-Z]+)',  # UPI/DR/123456/PARTYNAME
            r'PYTM[0-9]*/([A-Z]+)',  # PYTM + number + name
            r'SBIN[0-9]*/([A-Z]+)',  # SBIN reference
            r'UTIB[0-9]*/([A-Z]+)',  # UTIB reference
            r'ICIC[0-9]*/([A-Z]+)',  # ICIC reference
            r'TRANSFER\s+TO\s+([A-Z\s]+)',
            r'FROM\s+([A-Z\s]+)',
            r'PAID\s+TO\s+([A-Z\s]+)',
            r'RECEIVED\s+FROM\s+([A-Z\s]+)',
        ]
        
        for p in patterns:
            m = re.search(p, text)
            if m:
                party = self._clean_party(m.group(1))
                if party and party not in ["UNKNOWN", "PYTM", "SBIN", "UTIB", "ICIC"]:
                    return party
        
        # Fallback: extract meaningful words
        words = [
            w for w in text.split()
            if w.isalpha() and len(w) > 3 and w not in self.SKIP_WORDS
        ]
        
        return " ".join(words[:3]) if words else "UNKNOWN"
    
    def _clean_party(self, name: str) -> str:
        """Clean party name"""
        name = re.sub(r'[^A-Z\s]', '', name)
        return " ".join(name.split()).strip()
    
    def _clean_description(self, text: str) -> str:
        """Clean transaction description"""
        text = self.DATE_REGEX.sub("", text)
        text = re.sub(r'[₹$€£¥]\s*[\d,]+\.?\d*', '', text)
        text = re.sub(r'[\d,]+\.\d{2}\s*(?:Cr|Dr)?', '', text)
        return " ".join(text.split()).strip()
    
    def _normalize_date(self, raw: str) -> str:
        """Normalize date to DD/MM/YYYY"""
        raw = raw.replace("-", "/")
        d, m, y = raw.split("/")
        
        if len(y) == 2:
            y = "20" + y
        
        return f"{d.zfill(2)}/{m.zfill(2)}/{y}"
