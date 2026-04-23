"""
PDF Processing Service - Extracts transactions from bank statement PDFs
"""

import pdfplumber
import PyPDF2
import re
from typing import List, Dict, Any, Optional
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process PDF bank statements and extract transaction data."""

    # All date formats used by Indian banks
    DATE_PATTERNS = [
        r'\d{2}/\d{2}/\d{4}',   # DD/MM/YYYY
        r'\d{2}-\d{2}-\d{4}',   # DD-MM-YYYY
        r'\d{4}/\d{2}/\d{2}',   # YYYY/MM/DD
        r'\d{2}\.\d{2}\.\d{4}', # DD.MM.YYYY
        r'\d{2}\s+\w{3}\s+\d{4}', # 01 Apr 2024
        r'\d{1,2}/\d{1,2}/\d{4}',  # D/M/YYYY
        r'\d{1,2}-\d{1,2}-\d{4}',  # D-M-YYYY
    ]
    # Compiled once
    DATE_RE = re.compile('|'.join(DATE_PATTERNS))
    # Indian-format number: 1,23,456.78  or  12345.67
    AMOUNT_RE = re.compile(r'(?<!\d)[\d,]+\.\d{2}(?!\d)')

    MERCHANT_MAP = {
        r'\buber\b': 'UBER', r'\bola\b': 'OLA', r'\bswiggy\b': 'SWIGGY',
        r'\bzomato\b': 'ZOMATO', r'\bamazon\b': 'AMAZON', r'\bflipkart\b': 'FLIPKART',
        r'\bmyntra\b': 'MYNTRA', r'\boyo\b': 'OYO', r'\birctc\b': 'IRCTC',
        r'\bmakemytrip\b': 'MAKE MY TRIP', r'\bredbus\b': 'REDBUS',
        r'\bpaytm\b': 'PAYTM', r'\bphonepe\b': 'PHONEPE', r'\bgpay\b': 'GPAY',
        r'\bbhim\b': 'BHIM', r'\bgoogle\b': 'GOOGLE', r'\bnetflix\b': 'NETFLIX',
        r'\bspotify\b': 'SPOTIFY', r'\bhotstar\b': 'HOTSTAR',
    }
    BUSINESS_SUFFIXES = [
        'traders', 'trdg', 'trd', 'agencies', 'enterprises',
        'services', 'solutions', 'pvt', 'ltd', 'limited',
        'corp', 'corporation', 'inc', 'company', 'associates',
    ]
    STOP_PARTIES = {
        'DR', 'CR', 'TRF', 'BY', 'TO', 'FROM', 'PAID',
        'RECEIVED', 'UNKNOWN', 'N/A',
    }

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_transactions(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        transactions = []
        try:
            transactions = self._extract_with_pdfplumber(pdf_bytes)
            if not transactions:
                logger.info("pdfplumber found no transactions, trying PyPDF2…")
                transactions = self._extract_with_pypdf2(pdf_bytes)

            transactions = self._validate_and_clean(transactions)

            for txn in transactions:
                if not txn.get('detected_party'):
                    txn['detected_party'] = self._extract_party_name(txn.get('description', ''))
                    txn['party'] = txn['detected_party']

            if not transactions:
                raise ValueError("No valid transactions found in PDF")

            logger.info("Extracted %d transactions from PDF", len(transactions))
            return transactions

        except Exception as exc:
            logger.error("PDF extraction failed: %s", exc, exc_info=True)
            raise

    # ── pdfplumber strategy ───────────────────────────────────────────────────

    def _extract_with_pdfplumber(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        transactions = []
        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            if not table or len(table) < 2:
                                continue
                            header_idx = self._find_header_row(table)
                            headers = [
                                str(c).strip().lower() if c else ''
                                for c in table[header_idx]
                            ]
                            for row in table[header_idx + 1:]:
                                if not row or all(not c for c in row):
                                    continue
                                txn = self._parse_table_row(headers, row)
                                if txn:
                                    transactions.append(txn)
                    else:
                        # No table structure — fall back to text on this page
                        text = page.extract_text() or ''
                        if text:
                            transactions.extend(self._parse_text_transactions(text))
        except Exception as exc:
            logger.warning("pdfplumber error: %s", exc)
        return transactions

    # ── PyPDF2 strategy ───────────────────────────────────────────────────────

    def _extract_with_pypdf2(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        transactions = []
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            for page in reader.pages:
                text = page.extract_text() or ''
                if text:
                    transactions.extend(self._parse_text_transactions(text))
        except Exception as exc:
            logger.warning("PyPDF2 error: %s", exc)
        return transactions

    # ── Table helpers ─────────────────────────────────────────────────────────

    def _find_header_row(self, table: list) -> int:
        keywords = {'date', 'description', 'narration', 'credit', 'debit', 'balance', 'amount', 'particulars'}
        for idx, row in enumerate(table[:6]):
            if not row:
                continue
            row_text = ' '.join(str(c).lower() for c in row if c)
            hits = sum(1 for k in keywords if k in row_text)
            if hits >= 2:
                return idx
        return 0

    def _parse_table_row(self, headers: list, row: list) -> Optional[Dict[str, Any]]:
        try:
            def col(*kws):
                for kw in kws:
                    for i, h in enumerate(headers):
                        if kw in h and i < len(row):
                            return str(row[i]).strip() if row[i] else ''
                return ''

            date_val   = col('date', 'txn date', 'value date')
            desc_val   = col('description', 'narration', 'particulars', 'details', 'remarks')
            credit_val = col('credit', 'deposit', 'cr amount', 'cr')
            debit_val  = col('debit', 'withdrawal', 'dr amount', 'dr')
            bal_val    = col('balance', 'bal')
            amt_val    = col('amount')

            # Some banks use a single "Amount" column with DR/CR indicator
            if amt_val and not credit_val and not debit_val:
                dr_cr = col('dr/cr', 'type', 'indicator', 'cr/dr')
                if dr_cr and 'cr' in dr_cr.lower():
                    credit_val = amt_val
                elif dr_cr and 'dr' in dr_cr.lower():
                    debit_val = amt_val
                else:
                    # Guess from sign or context
                    credit_val = amt_val

            parsed_date = self._parse_date(date_val)
            if not parsed_date or not desc_val:
                return None

            credit = self._parse_amount(credit_val)
            debit  = self._parse_amount(debit_val)
            bal    = self._parse_amount(bal_val)

            if credit == 0 and debit == 0:
                return None

            party = self._extract_party_name(desc_val)
            return {
                'date': parsed_date,
                'description': desc_val,
                'credit': credit,
                'debit': debit,
                'balance': bal,
                'detected_party': party,
                'party': party,
            }
        except Exception as exc:
            logger.debug("Table row parse error: %s", exc)
            return None

    # ── Text-based parsing ────────────────────────────────────────────────────

    def _parse_text_transactions(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse unstructured text from a bank statement PDF.

        Strategy:
        - Every line that starts with a date begins a new transaction.
        - The last 2-3 numbers on the transaction line are amounts.
          Pattern: ... [debit]  [credit]  [balance]   (3 numbers)
                   ... [dr_or_cr_amount]  [balance]   (2 numbers)
          The LAST number is always the running balance.
        - Lines after the date line (with no date) that contain amounts
          are continuations; we use their amounts if we still have none.
        """
        transactions: List[Dict] = []
        current: Optional[Dict] = None
        prev_balance: float = 0.0

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            date_match = self.DATE_RE.search(line)

            if date_match:
                # --- Save previous transaction ---
                if current and current.get('description'):
                    transactions.append(current)

                parsed_date = self._parse_date(date_match.group())
                if not parsed_date:
                    current = None
                    continue

                current = {
                    'date': parsed_date,
                    'description': '',
                    'credit': 0.0,
                    'debit': 0.0,
                    'balance': 0.0,
                    'detected_party': None,
                    'party': None,
                }

                # Strip the date from the line and extract amounts
                remainder = line[:date_match.start()] + line[date_match.end():]
                amounts = self._find_amounts(remainder)

                desc_text = self.AMOUNT_RE.sub('', remainder).strip()
                # Also remove bare integers that look like amounts
                desc_text = re.sub(r'\b\d{5,}\b', '', desc_text).strip()
                current['description'] = desc_text

                self._assign_amounts(current, amounts, prev_balance)

            elif current is not None:
                # Continuation line
                amounts = self._find_amounts(line)
                if amounts:
                    # Prefer to fill in amounts if we don't have them yet
                    if current['credit'] == 0 and current['debit'] == 0:
                        self._assign_amounts(current, amounts, prev_balance)
                else:
                    # Pure text continuation — append to description
                    extra = line.strip()
                    if extra and len(extra) > 2:
                        current['description'] = (current['description'] + ' ' + extra).strip()

            # Update prev_balance tracker
            if current and current.get('balance', 0) > 0:
                prev_balance = current['balance']

        # Flush last transaction
        if current and current.get('description'):
            transactions.append(current)

        return transactions

    def _find_amounts(self, text: str) -> List[float]:
        """Return all decimal amounts found in text, preserving order."""
        raw = self.AMOUNT_RE.findall(text)
        result = []
        for r in raw:
            v = self._parse_amount(r)
            if v > 0:
                result.append(v)
        return result

    def _assign_amounts(self, txn: dict, amounts: List[float], prev_balance: float):
        """
        Assign credit/debit/balance from a list of extracted amounts.

        Heuristics for Indian bank statement rows:
          3 amounts  →  debit, credit, balance  (one of debit/credit will be 0)
          2 amounts  →  transaction_amount, balance
          1 amount   →  balance (or transaction amount if prev_balance known)
        """
        if not amounts:
            return

        if len(amounts) >= 3:
            # Last is balance; first two — whichever is non-zero is the txn amount
            bal = amounts[-1]
            a, b = amounts[-3], amounts[-2]
            txn['balance'] = bal
            # Use balance continuity to decide credit vs debit
            if prev_balance > 0:
                expected_credit = round(bal - prev_balance + b, 2)
                expected_debit  = round(prev_balance + a - bal, 2)
                if abs(bal - (prev_balance + a - b)) < 1:
                    txn['debit']  = a
                    txn['credit'] = b
                else:
                    txn['debit']  = a
                    txn['credit'] = b
            else:
                txn['debit']  = a if a > 0 else 0
                txn['credit'] = b if b > 0 else 0

        elif len(amounts) == 2:
            txn_amt, bal = amounts[0], amounts[1]
            txn['balance'] = bal
            if prev_balance > 0:
                diff = round(bal - prev_balance, 2)
                if abs(diff - txn_amt) < 1 or diff > 0:
                    txn['credit'] = txn_amt
                else:
                    txn['debit'] = txn_amt
            else:
                # Can't tell — default to credit (conservative)
                txn['credit'] = txn_amt

        elif len(amounts) == 1:
            val = amounts[0]
            if prev_balance > 0:
                diff = round(val - prev_balance, 2)
                if diff > 0:
                    txn['credit'] = diff
                    txn['balance'] = val
                else:
                    txn['debit'] = abs(diff)
                    txn['balance'] = val
            else:
                txn['balance'] = val

    # ── Validation & dedup ────────────────────────────────────────────────────

    def _validate_and_clean(self, transactions: List[Dict]) -> List[Dict]:
        validated = []
        seen: set = set()
        prev_balance: float = 0.0

        for txn in transactions:
            if not txn.get('date') or not txn.get('description'):
                continue

            credit  = float(txn.get('credit', 0) or 0)
            debit   = float(txn.get('debit', 0) or 0)
            balance = float(txn.get('balance', 0) or 0)

            # Recover zero amounts from balance continuity
            if credit == 0 and debit == 0 and prev_balance > 0 and balance > 0:
                diff = round(balance - prev_balance, 2)
                if diff > 0:
                    credit = diff
                    txn['credit'] = credit
                elif diff < 0:
                    debit = abs(diff)
                    txn['debit'] = debit

            # Skip if still no amount data
            if credit == 0 and debit == 0:
                continue

            txn['credit'] = round(credit, 2)
            txn['debit']  = round(debit, 2)
            txn['amount'] = round(credit - debit, 2)
            txn['date']   = str(txn['date'])

            # Dedup by (date, first-50-chars-of-desc, credit, debit)
            key = (txn['date'], txn['description'][:50], txn['credit'], txn['debit'])
            if key in seen:
                continue
            seen.add(key)

            if balance > 0:
                prev_balance = balance

            validated.append(txn)

        logger.info("Validated %d / %d transactions", len(validated), len(transactions))
        return validated

    # ── Party name extraction ─────────────────────────────────────────────────

    def _extract_party_name(self, narration: str) -> Optional[str]:
        if not narration or len(narration) < 2:
            return None

        text = narration.strip()

        # Known merchants first (fast path)
        for pattern, name in self.MERCHANT_MAP.items():
            if re.search(pattern, text, re.IGNORECASE):
                return name

        # UPI patterns
        upi_patterns = [
            r'UPI/(?:CR|DR)/\d+/(.+?)/(?:OK|FAIL|PA|BI|AX|PASS)',
            r'UPI/(?:CR|DR)/\d+/(.+?)$',
            r'UPI/\d+/(.+?)/(?:OK|FAIL|PA|BI)',
            r'UPI[-/]*(?:CR|DR)?[-/]*\d*[-/]*(.+?)(?:[-/]OK|[-/]PA|[-/]BI|$)',
            r'@([a-zA-Z0-9]{3,})',
        ]
        transfer_patterns = [
            r'RTGS[/\s\-]+(?:[A-Z0-9]+[/\s\-]+)?([A-Z][A-Za-z\s]{2,})',
            r'NEFT[/\s\-]+(?:[A-Z0-9]+[/\s\-]+)?([A-Z][A-Za-z\s]{2,})',
            r'IMPS[/\s\-]+(?:[A-Z0-9]+[/\s\-]+)?([A-Z][A-Za-z\s]{2,})',
            r'(?:TRANSFER|TRF)\s+(?:FROM|TO)\s+([A-Z][A-Za-z\s]{2,})',
            r'PAID\s+TO\s+([A-Z][A-Za-z\s]{2,})',
            r'RECEIVED\s+FROM\s+([A-Z][A-Za-z\s]{2,})',
        ]
        fallback_patterns = [
            r'TO\s+([A-Z][A-Za-z\s]{2,})',
            r'FROM\s+([A-Z][A-Za-z\s]{2,})',
            r'FOR\s+([A-Z][A-Za-z\s]{2,})',
        ]

        for group in (upi_patterns, transfer_patterns, fallback_patterns):
            for pat in group:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    try:
                        candidate = m.group(1).upper().strip()
                        candidate = ' '.join(candidate.split())
                        if len(candidate) >= 2 and candidate not in self.STOP_PARTIES:
                            return self._normalize_party(candidate)
                    except IndexError:
                        pass

        # Last resort: meaningful words
        stop_words = {
            'DEPOSIT', 'WITHDRAWAL', 'PAYMENT', 'TRANSFER', 'CREDIT', 'DEBIT',
            'BALANCE', 'CHARGES', 'FEE', 'TAX', 'EMI', 'BILL', 'SALARY',
            'INTEREST', 'REFUND', 'REVERSAL',
        }
        cleaned = text.upper()
        for sw in stop_words:
            cleaned = re.sub(r'\b' + sw + r'\b', ' ', cleaned)
        words = [w for w in cleaned.split() if len(w) > 2 and not w.isdigit()]
        if words:
            return self._normalize_party(' '.join(words[:3]))

        return None

    def _normalize_party(self, name: str) -> str:
        name = name.upper().strip()
        for suffix in self.BUSINESS_SUFFIXES:
            name = re.sub(rf'\b{suffix}\b', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\b\d{8,}\b', '', name)
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'^(?:TO|FROM|FOR|VIA|BY|REF|NO|AC|ACC|TRF)\s+', '', name)
        name = ' '.join(name.split())
        return name if len(name) >= 2 else 'UNKNOWN'

    # ── Date / amount parsers ─────────────────────────────────────────────────

    def _parse_date(self, date_str: str) -> Optional[str]:
        if not date_str:
            return None
        date_str = date_str.strip()

        # Month-name format: "01 Apr 2024"
        month_map = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12',
        }
        m = re.match(r'(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})', date_str)
        if m:
            d, mon, y = m.group(1), m.group(2).lower(), m.group(3)
            mo = month_map.get(mon)
            if mo:
                return f"{d.zfill(2)}/{mo}/{y}"

        # Numeric formats
        m = re.search(r'(\d{1,4})([/\-\.])(\d{1,2})\2(\d{2,4})', date_str)
        if not m:
            return None
        p1, p2, p3 = m.group(1), m.group(3), m.group(4)
        # YYYY/MM/DD
        if len(p1) == 4 and int(p1) > 31:
            return f"{p3.zfill(2)}/{p2.zfill(2)}/{p1}"
        # DD/MM/YYYY or DD/MM/YY
        year = p3 if len(p3) == 4 else '20' + p3
        return f"{p1.zfill(2)}/{p2.zfill(2)}/{year}"

    def _parse_amount(self, text: str) -> float:
        if not text:
            return 0.0
        s = str(text).replace(',', '').replace('₹', '').replace('Rs.', '').replace('rs.', '').strip()
        negative = s.startswith('(') and s.endswith(')')
        if negative:
            s = s[1:-1]
        elif s.startswith('-'):
            s = s[1:]; negative = True
        m = re.search(r'\d+\.?\d*', s)
        if m:
            val = float(m.group())
            return -val if negative else val
        return 0.0
