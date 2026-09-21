import logging
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)

class TableExtractionService:
    def __init__(self):
        try:
            import camelot
            self.enabled = True
        except ImportError:
            logger.warning("Camelot-py not installed. Table extraction disabled.")
            self.enabled = False

    def extract_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
            
        tables_data = []
        try:
            import camelot
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            if not tables or len(tables) == 0:
                tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
                
            for i, table in enumerate(tables):
                df = table.df
                tables_data.append({
                    "table_id": f"table_{i}",
                    "page": table.page,
                    "raw_json": df.to_json(orient='records'),
                    "rows": df.to_dict(orient='records')
                })
            return tables_data
        except Exception as e:
            logger.error(f"Table extraction failed for {pdf_path}: {e}")
            return []
