"""
AcuTrace - Party Ledger & Fund Flow Intelligence Platform
Backend API Server - Fixed Version with Proper Type Preservation
"""

import re
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import uvicorn
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

from services.excel_processor import ExcelProcessor
from services.pdf_processor import PDFProcessor
from services.entity_normalizer import EntityNormalizer
from services.fund_flow_chain_builder import FundFlowChainBuilder
from services.transaction_categorizer import TransactionCategorizer
from services.export_service import ExportService
from services.validation_engine import ValidationEngine
from services.balance_tracker import analyze_statement_balance

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AcuTrace API",
    description="Party Ledger & Fund Flow Intelligence Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

excel_processor = ExcelProcessor()
pdf_processor = PDFProcessor()
entity_normalizer = EntityNormalizer()
fund_flow_builder = FundFlowChainBuilder()
categorizer = TransactionCategorizer()
validation_engine = ValidationEngine()
export_service = ExportService()

SUPPORTED_EXTENSIONS = ('.xls', '.xlsx', '.pdf')


def _safe_get_numeric(value, default=0):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _calculate_amount(txn):
    """Calculate net amount from transaction"""
    amount = _safe_get_numeric(txn.get('amount'))
    if amount != 0:
        return amount
    credit = _safe_get_numeric(txn.get('credit'))
    debit = _safe_get_numeric(txn.get('debit'))
    if credit > 0 and debit == 0:
        return credit
    elif debit > 0 and credit == 0:
        return -debit
    elif credit > debit:
        return credit - debit
    elif debit > credit:
        return -(debit - credit)
    else:
        return 0


def _is_credit_transaction(txn):
    """Check if transaction is credit (money in)"""
    credit = _safe_get_numeric(txn.get('credit'))
    debit = _safe_get_numeric(txn.get('debit'))
    if credit > 0 and debit == 0:
        return True
    elif credit > debit:
        return True
    elif debit > credit:
        return False
    else:
        amount = _safe_get_numeric(txn.get('amount'))
        if amount != 0:
            return amount > 0
        return False


def _extract_party_from_narration(narration: str) -> Optional[str]:
    """Extract party name from narration"""
    if not narration or len(narration) < 2:
        return None
    
    narration = narration.upper().strip()
    
    # UPI patterns
    upi_match = re.search(r'UPI/\w+/([A-Z]+)', narration)
    if upi_match:
        return upi_match.group(1)
    
    handle_match = re.search(r'@([A-Z0-9]+)', narration)
    if handle_match:
        return handle_match.group(1)
    
    to_match = re.search(r'TO\s+([A-Z][A-Z\s]{2,})', narration)
    if to_match:
        return to_match.group(1).strip()
    
    from_match = re.search(r'FROM\s+([A-Z][A-Z\s]{2,})', narration)
    if from_match:
        return from_match.group(1).strip()
    
    return None


def _normalize_party(party: str) -> str:
    """Clean up party name"""
    if not party:
        return "UNKNOWN"
    party = re.sub(r'[^\w\s]', ' ', party)
    party = ' '.join(party.split()).strip()
    return party if len(party) >= 2 else "UNKNOWN"


def _extract_transactions_safe(processor_result) -> List[Dict[str, Any]]:
    """Safely extract transactions from processor result"""
    if processor_result is None:
        return []
    if isinstance(processor_result, list):
        return processor_result
    if isinstance(processor_result, tuple):
        for item in processor_result:
            if isinstance(item, list):
                return item
        return []
    return []


@app.get("/")
async def root():
    return {"message": "AcuTrace API", "status": "operational", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "acutrace"}

@app.post("/api/analyze")
async def analyze_statement(file: UploadFile = File(...)):
    """Analyze a single bank statement"""
    try:
        if not file.filename.lower().endswith(SUPPORTED_EXTENSIONS):
            raise HTTPException(status_code=400, detail=f"Only {', '.join(SUPPORTED_EXTENSIONS)} files are supported")
        
        logger.info(f"Processing file: {file.filename}")
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        transactions = []
        
        if file.filename.lower().endswith(('.xls', '.xlsx')):
            logger.info("Extracting from Excel...")
            result = excel_processor.extract_transactions(file_bytes, file.filename)
            transactions = _extract_transactions_safe(result)
        elif file.filename.lower().endswith('.pdf'):
            logger.info("Extracting from PDF...")
            result = pdf_processor.extract_transactions(file_bytes)
            transactions = _extract_transactions_safe(result)
            if not transactions:
                raise ValueError("No transactions found in PDF")
        
        if not transactions or len(transactions) == 0:
            raise HTTPException(status_code=400, detail="No transactions found")
        
        logger.info(f"Extracted {len(transactions)} transactions")
        
        # Process transactions
        entity_normalizer.clear()
        fund_flow_builder.clear()
        
        # Count credit/debit for logging
        pdf_credits = sum(1 for t in transactions if t.get('type') == 'CREDIT')
        pdf_debits = sum(1 for t in transactions if t.get('type') == 'DEBIT')
        logger.info(f"From PDF: {pdf_credits} CREDIT, {pdf_debits} DEBIT")
        
        for idx, txn in enumerate(transactions):
            try:
                # Preserve the original type from PDF!
                original_type = txn.get('type', 'UNKNOWN')
                
                txn['amount'] = _calculate_amount(txn)
                description = txn.get('description', '')
                is_credit = _is_credit_transaction(txn)
                amount = _safe_get_numeric(txn.get('amount'))
                
                # Extract party
                party = _extract_party_from_narration(description)
                if not party:
                    party = _normalize_party(description.split()[0] if description else '')
                
                if party and party != "UNKNOWN":
                    normalized = entity_normalizer._normalize_name(party)
                    if normalized:
                        entity_normalizer._register_entity(normalized, party, 'General', amount, is_credit, None)
                        txn['party'] = normalized
                        txn['detected_party'] = normalized
                else:
                    txn['party'] = "UNKNOWN"
                    txn['detected_party'] = "UNKNOWN"
                
                # Get category WITHOUT overwriting the type!
                category_data = categorizer.categorize_transaction(txn)
                
                # Update with category data BUT preserve original type!
                for key, value in category_data.items():
                    if key != 'type':  # Don't overwrite type from PDF!
                        txn[key] = value
                
                # Ensure type is preserved
                txn['type'] = original_type
                
            except Exception as e:
                logger.warning(f"Error processing transaction {idx}: {str(e)}")
                continue
        
        # Validation
        validated, validation_report = validation_engine.validate_transactions(transactions)
        transactions = validated
        
        # Balance analysis
        balance_summary = analyze_statement_balance(transactions)
        
        # Fund flow
        fund_flow_builder.add_transactions(transactions, file.filename)
        fund_flow_builder.build_chains()
        
        party_ledger = entity_normalizer.get_party_ledger_summary()
        fund_flow_chains = fund_flow_builder.get_chain_summary()
        entity_relations = entity_normalizer.get_entity_relation_index()
        
        # Final count
        final_credits = sum(1 for t in transactions if t.get('type') == 'CREDIT')
        final_debits = sum(1 for t in transactions if t.get('type') == 'DEBIT')
        logger.info(f"Final: {final_credits} CREDIT, {final_debits} DEBIT")
        
        source_type = "pdf" if file.filename.lower().endswith('.pdf') else "xls"
        
        return JSONResponse(content={
            "status": "success",
            "metadata": {
                "filename": file.filename,
                "total_transactions": len(transactions),
                "analysis_timestamp": datetime.now().isoformat(),
                "source": f"single_{source_type}",
                "credit_count": final_credits,
                "debit_count": final_debits
            },
            "transactions": transactions,
            "validation_report": validation_report.to_dict(),
            "balance_summary": balance_summary.to_dict(),
            "party_ledger": {
                "parties": party_ledger,
                "total_parties": len(party_ledger),
                "statistics": entity_normalizer.get_statistics()
            },
            "fund_flow_chains": fund_flow_chains,
            "entity_relations": entity_relations
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/api/analyze/multi")
async def analyze_multiple_statements(files: List[UploadFile] = File(...)):
    """Analyze multiple bank statement files"""
    try:
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        
        logger.info(f"Processing {len(files)} files simultaneously...")
        
        import asyncio
        
        async def process_file(file: UploadFile):
            try:
                if not file.filename.lower().endswith(SUPPORTED_EXTENSIONS):
                    return None, None, {}
                
                file_bytes = await file.read()
                if len(file_bytes) == 0:
                    return None, None, {}
                
                transactions = []
                
                if file.filename.lower().endswith(('.xls', '.xlsx')):
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, excel_processor.extract_transactions, file_bytes, file.filename
                    )
                    transactions = _extract_transactions_safe(result)
                elif file.filename.lower().endswith('.pdf'):
                    result = pdf_processor.extract_transactions(file_bytes)
                    transactions = _extract_transactions_safe(result)
                
                if not transactions:
                    return None, None, {}
                
                for txn in transactions:
                    txn['_source_file'] = file.filename
                
                metadata = {
                    "filename": file.filename,
                    "file_type": "pdf" if file.filename.lower().endswith('.pdf') else "xls",
                    "transaction_count": len(transactions)
                }
                
                return transactions, metadata, {}
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                return None, None, {}
        
        tasks = [process_file(file) for file in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_transactions = []
        file_metadata = []
        
        for result in results:
            if isinstance(result, Exception):
                continue
            transactions, metadata, _ = result
            if transactions and metadata:
                all_transactions.extend(transactions)
                file_metadata.append(metadata)
        
        if not all_transactions:
            raise HTTPException(status_code=400, detail="No transactions found in any file")
        
        logger.info(f"Total: {len(all_transactions)} transactions from {len(file_metadata)} files")
        
        entity_normalizer.clear()
        fund_flow_builder.clear()
        
        for txn in all_transactions:
            try:
                original_type = txn.get('type', 'UNKNOWN')
                
                txn['amount'] = _calculate_amount(txn)
                is_credit = _is_credit_transaction(txn)
                amount = _safe_get_numeric(txn.get('amount'))
                description = txn.get('description', '')
                
                party = _extract_party_from_narration(description)
                if not party:
                    party = _normalize_party(description.split()[0] if description else '')
                
                if party and party != "UNKNOWN":
                    normalized = entity_normalizer._normalize_name(party)
                    if normalized:
                        entity_normalizer._register_entity(normalized, party, 'General', amount, is_credit, None)
                        txn['party'] = normalized
                
                # Get category without overwriting type
                category_data = categorizer.categorize_transaction(txn)
                for key, value in category_data.items():
                    if key != 'type':
                        txn[key] = value
                
                # Preserve original type
                txn['type'] = original_type
                
            except Exception as e:
                logger.warning(f"Error: {str(e)}")
        
        # Validation
        validated, validation_report = validation_engine.validate_transactions(all_transactions)
        all_transactions = validated
        
        fund_flow_builder.add_transactions(all_transactions)
        fund_flow_builder.build_chains()
        
        party_ledger = entity_normalizer.get_party_ledger_summary()
        fund_flow_chains = fund_flow_builder.get_chain_summary()
        entity_relations = entity_normalizer.get_entity_relation_index()
        balance_summary = analyze_statement_balance(all_transactions)
        
        return JSONResponse(content={
            "status": "success",
            "metadata": {
                "files_processed": len(file_metadata),
                "file_details": file_metadata,
                "total_transactions": len(all_transactions),
                "analysis_timestamp": datetime.now().isoformat(),
                "source": "multi_file"
            },
            "transactions": all_transactions,
            "validation_report": validation_report.to_dict(),
            "balance_summary": balance_summary.to_dict(),
            "party_ledger": {
                "parties": party_ledger,
                "total_parties": len(party_ledger),
                "statistics": entity_normalizer.get_statistics()
            },
            "fund_flow_chains": fund_flow_chains,
            "entity_relations": entity_relations
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/api/party/{party_name}")
async def get_party_details(party_name: str):
    try:
        normalized = entity_normalizer._normalize_name(party_name)
        if normalized not in entity_normalizer.entities:
            raise HTTPException(status_code=404, detail=f"Party '{party_name}' not found")
        
        entity_data = entity_normalizer.entities[normalized]
        
        return JSONResponse(content={
            "status": "success",
            "party": {
                "name": normalized,
                "transaction_count": entity_data["transaction_count"],
                "total_credit": entity_data["total_credit"],
                "total_debit": entity_data["total_debit"],
                "net_flow": entity_data["total_credit"] - entity_data["total_debit"],
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/api/fund-flow/chains")
async def get_fund_flow_chains():
    try:
        chains = fund_flow_builder.get_chain_summary()
        return JSONResponse(content={"status": "success", "fund_flow_chains": chains})
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/api/party-ledger")
async def get_party_ledger():
    try:
        party_ledger = entity_normalizer.get_party_ledger_summary()
        statistics = entity_normalizer.get_statistics()
        return JSONResponse(content={"status": "success", "party_ledger": {"parties": party_ledger, "total_parties": len(party_ledger), "statistics": statistics}})
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
