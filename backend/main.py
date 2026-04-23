"""
AcuTrace - Party Ledger & Fund Flow Intelligence Platform
Backend API Server
"""

import re
import os
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

from auth import UserCreate, Token, get_current_user, register_user, login_user
from services.excel_processor import ExcelProcessor
from services.entity_normalizer import EntityNormalizer
from services.fund_flow_chain_builder import FundFlowChainBuilder
from services.transaction_categorizer import TransactionCategorizer
from services.export_service import ExportService

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── CORS origins ──────────────────────────────────────────────────────────────
_raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip() and o.strip() != "*"]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:5173"]

# Regex covers every Vercel preview URL (*.vercel.app) + localhost variants
CORS_ORIGIN_REGEX = os.getenv(
    "CORS_ORIGIN_REGEX",
    r"https://.*\.vercel\.app|http://localhost:\d+|http://127\.0\.0\.1:\d+"
)

# ── File upload limits ────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", "25")) * 1024 * 1024
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", "10"))
SUPPORTED_EXTENSIONS = (".xls", ".xlsx")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AcuTrace API",
    description="Party Ledger & Fund Flow Intelligence Platform",
    version="2.0.0",
    docs_url=None,   # disable public Swagger UI in production
    redoc_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
)

# ── Service instances ─────────────────────────────────────────────────────────
excel_processor = ExcelProcessor()
entity_normalizer = EntityNormalizer()
fund_flow_builder = FundFlowChainBuilder()
categorizer = TransactionCategorizer()
export_service = ExportService()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile, file_bytes: bytes) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File has no name")
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}",
        )
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size ({MAX_FILE_SIZE_BYTES // (1024*1024)} MB)",
        )


def _extract_party_from_narration(narration: str) -> Optional[str]:
    if not narration or len(narration) < 2:
        return None

    narration_upper = narration.upper().strip()
    party = None

    upi_patterns = [
        r"UPI/(?:CR|DR)/\d+/(.+?)/(?:OK|FAIL|PA|BI|AX|PASS)",
        r"UPI/\d+/(.+?)/(?:OK|FAIL|PA|BI)$",
        r"UPI-(?:CR|DR)?-?\d*-?(.+?)(?:[-/]OK|[-/]FAIL|[-/]PA|[-/]BI|$)",
        r"@([a-zA-Z0-9]+)",
        r"UPI[/\s]*(?:from|to|by)[/\s]*([A-Z][A-Za-z\s]{2,})",
        r"UPI/(?:D\d+)?[/\s]*([A-Z][A-Za-z\s]{2,})",
        r"UPI[/\s]*(?:CR|DR)[/\s]*(?:D\d+)?[/\s]*([A-Z][A-Za-z\s]{2,})",
        r"(?:UPI|PAYTM|GPAY|PHONEPE)[/\s]*(?:CR|DR)?[/\s]*(?:D\d+)?[/\s]*([A-Z][A-Za-z\s]+?)(?:/OKPA|/OKAX|/OKBI|/OK)",
    ]
    for pattern in upi_patterns:
        m = re.search(pattern, narration_upper, re.IGNORECASE)
        if m and m.group(1):
            candidate = " ".join(m.group(1).upper().strip().split())
            if len(candidate) >= 2 and candidate not in {"DR", "CR", "TRF", "BY", "TO", "FROM"}:
                party = candidate
                break

    transfer_patterns = [
        r"(?:transfer|TRANSFER)\s+(?:from|to|FROM|TO)\s+([A-Z][A-Za-z\s]{2,})",
        r"PAID\s+TO\s+([A-Z][A-Za-z\s]{2,})",
        r"RECEIVED\s+FROM\s+([A-Z][A-Za-z\s]{2,})",
        r"BY\s+(?:TRANSFER|NEFT|RTGS|IMPS)[:\s-]*([A-Z][A-Za-z\s]{2,})",
        r"TRF\s+(?:TO|FROM)[:\s]*([A-Z][A-Za-z\s]{2,})",
    ]
    if not party:
        for pattern in transfer_patterns:
            m = re.search(pattern, narration_upper, re.IGNORECASE)
            if m and m.group(1):
                candidate = " ".join(m.group(1).upper().strip().split())
                if len(candidate) >= 2:
                    party = candidate
                    break

    other_transfer_patterns = [
        r"RTGS\s+(?:CR|DR)?[-]?\s*(?:[A-Z0-9]+[-])?\s*([A-Z][A-Za-z\s]{2,})",
        r"NEFT\s+(?:CR|DR)?[-]?\s*(?:[A-Z0-9]+[-])?\s*([A-Z][A-Za-z\s]{2,})",
        r"IMPS\s+(?:CR|DR)?[-]?\s*(?:[A-Z0-9]+[-])?\s*([A-Z][A-Za-z\s]{2,})",
    ]
    if not party:
        for pattern in other_transfer_patterns:
            m = re.search(pattern, narration_upper, re.IGNORECASE)
            if m and m.group(1):
                candidate = " ".join(m.group(1).upper().strip().split())
                if len(candidate) >= 2:
                    party = candidate
                    break

    other_patterns = [
        r"CASH\s+(?:DEPOSIT|WITHDRAWAL)\s*(?:AT|BY)?\s*([A-Z][A-Za-z\s]{2,})",
        r"(?:BILL|EMI|LOAN)\s+(?:PAYMENT|REPAYMENT)[:\s]*([A-Z][A-Za-z\s]{2,})",
        r"INSURANCE\s+(?:PREMIUM|PAYMENT)[:\s]*([A-Z][A-Za-z\s]{2,})",
        r"SALARY\s+(?:FROM|TO)?\s*([A-Z][A-Za-z\s]{2,})",
    ]
    if not party:
        for pattern in other_patterns:
            m = re.search(pattern, narration_upper, re.IGNORECASE)
            if m and m.group(1):
                candidate = " ".join(m.group(1).upper().strip().split())
                if len(candidate) >= 2:
                    party = candidate
                    break

    to_from_patterns = [
        r"TO\s+([A-Z][A-Za-z\s]{2,})",
        r"FOR\s+([A-Z][A-Za-z\s]{2,})",
        r"FROM\s+([A-Z][A-Za-z\s]{2,})",
        r"AT\s+([A-Z][A-Za-z\s]{2,})",
    ]
    if not party:
        for pattern in to_from_patterns:
            m = re.search(pattern, narration_upper, re.IGNORECASE)
            if m and m.group(1):
                candidate = m.group(1).upper().strip()
                candidate = re.sub(r"^(TO|FROM|FOR|AT|ON|BY|REF|NO|NEW|AC|ACC)\s*", "", candidate)
                candidate = " ".join(candidate.split())
                if len(candidate) >= 2:
                    party = candidate
                    break

    if party:
        suffixes = [
            "TRADERS", "TRDG", "AGENCIES", "SERVICES", "PVT", "LTD", "LIMITED",
            "CORP", "INC", "COMPANY", "HOLDINGS", "INDUSTRIES",
        ]
        for s in suffixes:
            party = re.sub(rf"\b{s}\b", "", party, flags=re.IGNORECASE)
        party = re.sub(r"[^\w\s]", " ", party)
        party = re.sub(r"\b\d{10,}\b", "", party)
        party = " ".join(party.split()).strip()
        if len(party) >= 2:
            return party

    if not party:
        STOP = {
            "DEPOSIT", "WITHDRAWAL", "PAYMENT", "TRANSFER", "CREDIT", "DEBIT",
            "BALANCE", "CHARGES", "FEE", "TAX", "EMI", "BILL", "SALARY",
            "INTEREST", "DIVIDEND", "REFUND", "REVERSAL", "CLEARING", "NO", "NUM",
            "BY", "TO", "FROM", "FOR", "AT", "ON",
        }
        cleaned = narration_upper
        for word in STOP:
            cleaned = re.sub(r"\b" + word + r"\b", " ", cleaned)
        words = cleaned.strip().split()
        meaningful = [w for w in words if len(w) > 2 and not w.isdigit()]
        if meaningful:
            candidate = " ".join(meaningful[:3]).upper()
            candidate = re.sub(r"[^\w\s]", " ", candidate)
            candidate = " ".join(candidate.split())
            if len(candidate) >= 2:
                return candidate

    return None


def _process_transactions(transactions: list, filename: str = "") -> dict:
    """Categorise and register parties for a list of transactions."""
    stats = {"total": len(transactions), "found": 0, "fallback": 0}
    GENERIC = {"DEPOSIT", "CASH", "WITHDRAWAL", "TRANSFER", "UNKNOWN", ""}

    for idx, txn in enumerate(transactions):
        try:
            if txn.get("amount", 0) == 0:
                txn["amount"] = (txn.get("credit", 0) or 0) - (txn.get("debit", 0) or 0)

            existing_party = txn.get("detected_party") or txn.get("party")
            description = txn.get("description", "")
            is_credit = txn.get("credit", 0) > 0
            amount = txn.get("amount", 0)

            party_to_register = existing_party
            if not party_to_register or party_to_register in GENERIC:
                fallback = _extract_party_from_narration(description)
                if fallback:
                    party_to_register = fallback
                    txn["detected_party"] = fallback
                    txn["party"] = fallback
                    stats["fallback"] += 1
                else:
                    words = (
                        description.replace("DEPOSIT", "")
                        .replace("PAYMENT", "")
                        .replace("CASH", "")
                        .replace("UTR", "")
                        .strip()
                        .split()
                    )
                    meaningful = [w for w in words if len(w) > 2 and not w.isdigit()]
                    if meaningful:
                        party_to_register = " ".join(meaningful[:3]).upper()
                        txn["detected_party"] = party_to_register
                        txn["party"] = party_to_register
                        stats["fallback"] += 1

            if party_to_register and party_to_register not in GENERIC:
                stats["found"] += 1
                entity_normalizer.extract_entity(description, amount, is_credit=is_credit)
                registered = entity_normalizer._normalize_name(party_to_register)
                if registered and registered in entity_normalizer.entities:
                    txn["party"] = registered
                    txn["detected_party"] = registered

            category_data = categorizer.categorize_transaction(txn)
            txn.update(category_data)
        except Exception as exc:
            logger.warning("Error processing transaction %d: %s", idx, exc)

    return stats


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=Token, tags=["auth"])
@limiter.limit("5/minute")
async def register(request: Request, data: UserCreate):
    """Register a new user and receive a JWT access token."""
    return register_user(data)


@app.post("/auth/login", response_model=Token, tags=["auth"])
@limiter.limit("10/minute")
async def login(request: Request, data: UserCreate):
    """Authenticate and receive a JWT access token."""
    return login_user(data)


# ── Public health endpoints ───────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def root():
    return {"message": "AcuTrace API", "status": "operational", "version": "2.0.0"}


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "healthy", "service": "acutrace-party-ledger"}


# ── Protected analysis endpoints ──────────────────────────────────────────────

@app.post("/api/analyze", tags=["analysis"])
@limiter.limit("10/minute")
async def analyze_statement(
    request: Request,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Analyze a single bank statement."""
    try:
        file_bytes = await file.read()
        _validate_upload(file, file_bytes)

        logger.info("User '%s' processing file: %s", current_user["username"], file.filename)

        transactions: list = []
        account_profile: dict = {}

        result = excel_processor.extract_transactions(file_bytes, file.filename)
        if isinstance(result, tuple) and len(result) == 2:
            transactions, account_profile = result
        else:
            transactions = result if isinstance(result, list) else []

        if not transactions:
            raise HTTPException(status_code=400, detail="No transactions found in the uploaded file")

        entity_normalizer.clear()
        fund_flow_builder.clear()

        stats = _process_transactions(transactions, file.filename)
        logger.info("Party extraction stats: %s", stats)

        fund_flow_builder.add_transactions(transactions, file.filename)
        fund_flow_builder.build_chains()

        party_ledger = entity_normalizer.get_party_ledger_summary()
        fund_flow_chains = fund_flow_builder.get_chain_summary()
        entity_relations = entity_normalizer.get_entity_relation_index()

        return JSONResponse(content={
            "status": "success",
            "metadata": {
                "filename": file.filename,
                "total_transactions": len(transactions),
                "analysis_timestamp": datetime.now().isoformat(),
                "source": "single_file",
                "party_extraction_stats": stats,
            },
            "transactions": transactions,
            "account_profile": account_profile,
            "party_ledger": {
                "parties": party_ledger,
                "total_parties": len(party_ledger),
                "statistics": entity_normalizer.get_statistics(),
            },
            "fund_flow_chains": fund_flow_chains,
            "entity_relations": entity_relations,
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error processing file: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis failed. Please check your file and try again.")


@app.post("/api/analyze/multi", tags=["analysis"])
@limiter.limit("5/minute")
async def analyze_multiple_statements(
    request: Request,
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Analyze multiple bank statement files simultaneously."""
    import asyncio

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum allowed is {MAX_FILES_PER_REQUEST}",
        )

    logger.info("User '%s' processing %d files", current_user["username"], len(files))

    async def process_file(file: UploadFile):
        try:
            file_bytes = await file.read()
            _validate_upload(file, file_bytes)

            transactions: list = []
            account_profile: dict = {}

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, excel_processor.extract_transactions, file_bytes, file.filename
            )
            if isinstance(result, tuple) and len(result) == 2:
                transactions, account_profile = result
            else:
                transactions = result if isinstance(result, list) else []

            if not transactions:
                return None, None, {}

            for txn in transactions:
                txn["_source_file"] = file.filename

            metadata = {
                "filename": file.filename,
                "file_type": "xls",
                "transaction_count": len(transactions),
            }
            return transactions, metadata, account_profile

        except HTTPException as exc:
            logger.warning("Skipping %s: %s", file.filename, exc.detail)
            return None, None, {}
        except Exception as exc:
            logger.error("Error processing %s: %s", file.filename, exc)
            return None, None, {}

    results = await asyncio.gather(*[process_file(f) for f in files], return_exceptions=True)

    all_transactions: list = []
    file_metadata: list = []
    combined_profile: dict = {}

    for result in results:
        if isinstance(result, Exception):
            continue
        txns, meta, profile = result
        if txns and meta:
            all_transactions.extend(txns)
            file_metadata.append(meta)
            for k, v in profile.items():
                if k not in combined_profile or not combined_profile[k]:
                    combined_profile[k] = v

    if not all_transactions:
        raise HTTPException(status_code=400, detail="No transactions found in any of the uploaded files")

    logger.info("Total extracted: %d transactions from %d files", len(all_transactions), len(file_metadata))

    entity_normalizer.clear()
    fund_flow_builder.clear()

    stats = _process_transactions(all_transactions)
    logger.info("Party extraction stats (multi): %s", stats)

    fund_flow_builder.add_transactions(all_transactions)
    fund_flow_builder.build_chains()
    merged = entity_normalizer.auto_merge_similar_entities()

    party_ledger = entity_normalizer.get_party_ledger_summary()
    fund_flow_chains = fund_flow_builder.get_chain_summary()
    entity_relations = entity_normalizer.get_entity_relation_index()

    return JSONResponse(content={
        "status": "success",
        "metadata": {
            "files_processed": len(file_metadata),
            "file_details": file_metadata,
            "total_transactions": len(all_transactions),
            "analysis_timestamp": datetime.now().isoformat(),
            "source": "multi_file",
            "auto_merged_entities": merged,
            "party_extraction_stats": stats,
        },
        "transactions": all_transactions,
        "account_profile": combined_profile,
        "party_ledger": {
            "parties": party_ledger,
            "total_parties": len(party_ledger),
            "statistics": entity_normalizer.get_statistics(),
        },
        "fund_flow_chains": fund_flow_chains,
        "entity_relations": entity_relations,
    })


@app.get("/api/party/{party_name}", tags=["data"])
async def get_party_details(
    party_name: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        normalized = entity_normalizer._normalize_name(party_name)
        if normalized not in entity_normalizer.entities:
            raise HTTPException(status_code=404, detail="Party not found")

        entity_data = entity_normalizer.entities[normalized]
        money_paths = fund_flow_builder.get_money_path_by_party(party_name)

        return JSONResponse(content={
            "status": "success",
            "party": {
                "name": normalized,
                "entity_type": entity_data.get("entity_type", "Unknown"),
                "transaction_count": entity_data["transaction_count"],
                "total_credit": entity_data["total_credit"],
                "total_debit": entity_data["total_debit"],
                "net_flow": entity_data["total_credit"] - entity_data["total_debit"],
                "upi_handles": list(entity_data.get("upi_handles", [])),
                "money_paths": money_paths,
            },
        })
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting party details: %s", exc)
        raise HTTPException(status_code=500, detail="Could not retrieve party details")


@app.get("/api/fund-flow/chains", tags=["data"])
async def get_fund_flow_chains(current_user: dict = Depends(get_current_user)):
    try:
        chains = fund_flow_builder.get_chain_summary()
        return JSONResponse(content={"status": "success", "fund_flow_chains": chains})
    except Exception as exc:
        logger.error("Error getting fund flow chains: %s", exc)
        raise HTTPException(status_code=500, detail="Could not retrieve fund flow chains")


@app.get("/api/party-ledger", tags=["data"])
async def get_party_ledger(current_user: dict = Depends(get_current_user)):
    try:
        party_ledger = entity_normalizer.get_party_ledger_summary()
        statistics = entity_normalizer.get_statistics()
        return JSONResponse(content={
            "status": "success",
            "party_ledger": {
                "parties": party_ledger,
                "total_parties": len(party_ledger),
                "statistics": statistics,
            },
        })
    except Exception as exc:
        logger.error("Error getting party ledger: %s", exc)
        raise HTTPException(status_code=500, detail="Could not retrieve party ledger")


@app.get("/api/relations", tags=["data"])
async def get_party_relations(current_user: dict = Depends(get_current_user)):
    try:
        relations = entity_normalizer.get_entity_relation_index()
        return JSONResponse(content={
            "status": "success",
            "relations": relations,
            "total_relations": len(relations),
        })
    except Exception as exc:
        logger.error("Error getting relations: %s", exc)
        raise HTTPException(status_code=500, detail="Could not retrieve party relations")


@app.post("/api/export/json", tags=["export"])
async def export_analysis(
    format: str = Query("json"),
    current_user: dict = Depends(get_current_user),
):
    try:
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "party_ledger": entity_normalizer.get_party_ledger_summary(),
            "fund_flow_chains": fund_flow_builder.get_chain_summary(),
            "entity_relations": entity_normalizer.get_entity_relation_index(),
        }
        return JSONResponse(content=export_data)
    except Exception as exc:
        logger.error("Error exporting data: %s", exc)
        raise HTTPException(status_code=500, detail="Export failed")


import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
