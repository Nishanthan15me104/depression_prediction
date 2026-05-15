import os
import logging
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

logger = logging.getLogger(__name__)

# Path to the reference data you will save
REFERENCE_DATA_PATH = "data/processed/reference_data.csv"

def generate_drift_report(current_data_buffer):
    """
    Compares the in-memory buffer against the reference CSV.
    Returns the HTML string of the Evidently report.
    """
    if len(current_data_buffer) < 5:
        return "<h3>Not enough data yet. Send at least 5 predictions first.</h3>"

    if not os.path.exists(REFERENCE_DATA_PATH):
        return f"<h3>Error: Reference data not found at {REFERENCE_DATA_PATH}</h3>"

    try:
        # 1. Load Reference Data
        ref_df = pd.read_csv(REFERENCE_DATA_PATH)
        
        # 2. Convert Buffer to DataFrame
        curr_df = pd.DataFrame(list(current_data_buffer))

        # 3. Initialize and Run Evidently Report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=curr_df)
        
        # 4. Return as HTML string to be rendered by FastAPI
        return report.get_html()

    except Exception as e:
        logger.error(f"Error generating drift report: {e}")
        return f"<h3>Error generating report: {str(e)}</h3>"