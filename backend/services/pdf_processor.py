"""
Universal Bank Statement PDF Analyzer
With DEBUG Logging for Credit/Debit Detection
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from io import BytesIO

import pdfplumber
import PyPDF2

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pdf_processor")

# Create a handler that prints to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


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
    
    logger.debug(f"Found {len(amount_with_suffix)} amounts with suffix: {amount_with_suffix}")
    
    for amount_str, suffix in amount_with_suffix:
        cleaned = re.sub(r'[,\s]', '', amount_str)
        try:
            amount = float(cleaned)
            if amount > 0:
                results.append((amount, suffix.upper()))
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
    
    logger.debug(f"Detecting type for: {text[:100]}...")
    
    # ==== 1. CHECK AMOUNT SUFFIX FIRST (most reliable) ====
    amounts_with_suffix = extract_amounts_from_text(text)
    
    # Separate credit and debit amounts
    credit_amounts = [a for a, s in amounts_with_suffix if s == 'CR']
    debit_amounts = [a for a, s in amounts_with_suffix if s == 'DR']
    
    logger.debug(f"Credit amounts: {credit_amounts}")
    logger.debug(f"Debit amounts: {debit_amounts}")
    
    # If we found credit amounts, this is a credit transaction
    if credit_amounts:
        txn_type = "CREDIT"
        confidence = 3.0
        reasons.append(f"amount suffix Cr: {credit_amounts[0]}")
        logger.debug(f"DETECTED: CREDIT (from Cr suffix)")
        return txn_type, confidence, reasons
    
    # If we found debit amounts, this is a debit transaction
    if debit_amounts:
        txn_type = "DEBIT"
        confidence = 3.0
        reasons.append(f"amount suffix Dr: {debit_amounts[0]}")
        logger.debug(f"DETECTED: DEBIT (from Dr suffix)")
        return txn_type, confidence, reasons
    
    # ==== 2. CHECK CR/DR PATTERNS ====
    cr_patterns = [r'\bCR\b', r'/CR/', r'\bCr\.\b']
    dr_patterns = [r'\bDR\b', r'/DR/', r'\bDr\.\b']
    
    cr_positions = []
    dr_positions = []
    
    for pattern in cr_patterns:
        for m in re.finditer(pattern, text_upper):
            cr_positions.append(m.start())
    
    for pattern in dr_patterns:
        for m in re.finditer(pattern, text_upper):
            dr_positions.append(m.start())
    
    last_cr = max(cr_positions) if cr_positions else -1
    last_dr = max(dr_positions) if dr_positions else -1
    
    logger.debug(f"CR positions: {cr_positions}, last_cr: {last_cr}")
    logger.debug(f"DR positions: {dr_positions}, last_dr: {last_dr}")
    
    if last_cr > last_dr:
        reasons.append(f"CR at position {last_cr}")
        logger.debug(f"DETECTED: CREDIT (CR at pos {last_cr})")
        return "CREDIT", 2.5, reasons
    elif last_dr > last_cr:
        reasons.append(f"DR at position {last_dr}")
        logger.debug(f"DETECTED: DEBIT (DR at pos {last_dr})")
        return "DEBIT", 2.5, reasons
    
    # ==== 3. CHECK KEYWORDS ====
    credit_keywords = [
        (r'SALARY', 2.0), (r'DEPOSIT', 2.0), (r'RECEIVED', 2.0),
        (r'REFUND', 2.0), (r'INTEREST', 2.0), (r'DIVIDEND', 2.0),
    ]
    
    debit_keywords = [
        (r'PAID', 2.0), (r'PAID\s+TO', 2.5), (r'WITHDRAWAL', 2.0),
        (r'\sWDL\s', 2.0), (r'TRANSFER\s+TO', 2.0), (r'EMI', 2.0),
    ]
    
    credit_signals = 0.0
    debit_signals = 0.0
    
    for pattern, weight in credit_keywords:
        if re.search(pattern, text_upper):
            credit_signals += weight
            reasons.append(f"credit: {pattern}")
    
    for pattern, weight in debit_keywords:
        if re.search(pattern, text_upper):
            debit_signals += weight
            reasons.append(f"debit: {pattern}")
    
    if credit_signals > debit_signals:
        logger.debug(f"DETECTED: CREDIT (keywords, score {credit_signals})")
        return "CREDIT", credit_signals, reasons
    elif debit_signals > credit_signals:
        logger.debug(f"DETECTED: DEBIT (keywords, score {debit_signals})")
        return "DEBIT", debit_signals, reasons
    
    # ==== 4. DEFAULT ====
    reasons.append("default to DEBIT")
    logger.debug("DETECTED: DEBIT (default)")
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
        logger.info("Starting PDF extraction...")
        
        text = self._extract_text(pdf_bytes)
        
        if not text.strip():
            logger.warning("No text extracted from PDF")
            return []
        
        logger.info(f"Extracted {len(text)} characters, {len(text.splitlines())} lines")
        
        transactions = self._parse_transactions(text)
        
        logger.info(f"Parsed {len(transactions)} transactions")
        
        # Log summary
        credits = sum(1 for t in transactions if t.get('type') == 'CREDIT')
        debits = sum(1 for t in transactions if t.get('type') == 'DEBIT')
        logger.info(f"Summary: {credits} CREDIT, {debits} DEBIT")
        
        return transactions
    
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
        
        # Assign amounts based on detected type
        debit = 0.0
        credit = 0.0
        balance = 0.0
        
        if txn_type == "CREDIT":
            if credit_amounts:
                credit = credit_amounts[0]
            elif amounts_with_suffix:
                credit = amounts_with_suffix[0][0]
        else:
            if debit_amounts:
                debit = debit_amounts[0]
            elif amounts_with_suffix:
                debit = amounts_with_suffix[0][0]
        
        # Balance is typically the last amount
        if len(amounts_with_suffix) >= 2:
            balance = amounts_with_suffix[-1][0]
        
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
        
        # Try various patterns
        patterns = [
            r'UPI/\w+/([A-Z]+)',
            r'PYTM[0-9]*/([A-Z]+)',
            r'SBIN[0-9]*/([A-Z]+)',
            r'UTIB[0-9]*/([A-Z]+)',
            r'TRANSFER\s+TO\s+([A-Z\s]+)',
            r'FROM\s+([A-Z\s]+)',
        ]
        
        for p in patterns:
            m = re.search(p, text)
            if m:
                party = self._clean_party(m.group(1))
                if party and party not in ["UNKNOWN", "PYTM", "SBIN", "UTIB"]:
                    return party
        
        words = [w for w in text.split() if w.isalpha() and len(w) > 3 and w not in self.SKIP_WORDS]
        return " ".join(words[:3]) if words else "UNKNOWN"
    
    def _clean_party(self, name: str) -> str:
        name = re.sub(r'[^A-Z\s]', '', name)
        return " ".join(name.split()).strip()
    
    def _clean_description(self, text: str) -> str:
        text = self.DATE_REGEX.sub("", text)
        text = re.sub(r'[₹$€£¥]\s*[\d,]+\.?\d*', '', text)
        text = re.sub(r'[\d,]+\.\d{2}\s*(?:Cr|Dr)?', '', text)
        return " ".join(text.split()).strip()
    
    def _normalize_date(self, raw: str) -> str:
        raw = raw.replace("-", "/")
        d, m, y = raw.split("/")
        if len(y) == 2:
            y = "20" + y
        return f"{d.zfill(2)}/{m.zfill(2)}/{y}"
