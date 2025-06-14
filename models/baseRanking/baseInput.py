from google.cloud import bigquery
import pandas as pd
import numpy as np
from datetime import datetime
import json
from google.oauth2 import service_account
import os
import joblib
from sqlalchemy import create_engine
import numpy as np
from datetime import datetime
import json

# Database connection setup
DATABASE_USER = "postgres"
DATABASE_PASSWORD = "oy159753"
DATABASE_HOST = "localhost"
DATABASE_PORT = "5432"
DATABASE_NAME = "university_db"
DATABASE_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

# Updated economic indicators for 2025
ECONOMIC_INDICATORS_2025 = {
    'base_salary_by_year': 22104,
    'inflation_by_year': 35.41,
    'growth_by_year': 2
}

# JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return json.JSONEncoder.default(self, obj)

def convert_value(value):
    """Convert any value to JSON-safe type"""
    if pd.isna(value):
        return None
    elif isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (int, float, str, bool)):
        return value
    else:
        # Try to convert using item() if available
        try:
            return value.item()
        except:
            return value

def get_bigquery_client(service_account_path):
    """Returns an authenticated BigQuery client using service account credentials."""
    credentials = service_account.Credentials.from_service_account_file(
        service_account_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return bigquery.Client(credentials=credentials)

def create_base_input_table(project_id, dataset_id, table_id, service_account_path):
    """Creates the BigQuery table for storing base input vectors."""
    client = get_bigquery_client(service_account_path)
    
    schema = [
        bigquery.SchemaField("idOSYM", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("academicYear", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("base_input", "STRING", mode="REQUIRED"),  # JSON as STRING for file loading
        bigquery.SchemaField("last_updated", "TIMESTAMP", mode="REQUIRED"),
    ]
    
    # Create dataset if it doesn't exist
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        client.create_dataset(dataset)
        print(f"Created dataset {dataset_id}")
    
    # Create table
    table_ref = dataset_ref.table(table_id)
    table = bigquery.Table(table_ref, schema=schema)
    table = client.create_table(table, exists_ok=True)
    print(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")

class BaseInputGenerator:
    def __init__(self, project_id, dataset_id, table_id, service_account_path, all_unique_features):
        """
        Initialize the base input generator with PostgreSQL connection.
        
        Args:
            project_id: BigQuery project ID
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            service_account_path: Path to service account JSON (optional)
        """
        self.engine = create_engine(DATABASE_URL)
        self.all_unique_features = all_unique_features
        if service_account_path:
            self.client = get_bigquery_client(service_account_path)
        else:
            self.client = bigquery.Client(project=project_id)
            
        self.table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    def _convert_pandas_to_dict(self, pandas_series):
        """Convert pandas Series to dictionary with proper type conversion"""
        result = {}
        for key, value in pandas_series.items():
            result[key] = convert_value(value)
        return result
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(v) for v in obj]
        else:
            return convert_value(obj)
        
    def _get_historical_data(self, program_id=None):
        """Load historical data from PostgreSQL."""
        table_name = "encoded_data_avg_vakıf_os_model"
        if program_id:
            query = f"""
                SELECT * FROM {table_name} 
                WHERE "idOSYM" = {program_id}
                ORDER BY "current_academicYear" DESC
            """
        else:
            query = f"SELECT * FROM {table_name}"
        return pd.read_sql(query, self.engine)

    def _get_raw_data(self, program_id=None):
        """Load raw data from PostgreSQL."""
        table_name = "exterpolated"
        if program_id:
            query = f"""
                SELECT * FROM {table_name} 
                WHERE "idOSYM" = {program_id}
                ORDER BY "academicYear" DESC
            """
        else:
            query = f"SELECT * FROM {table_name}"
        return pd.read_sql(query, self.engine)

    def _get_base_input_vector(self, program_id, prediction_year):
        """Generate base input vector using PostgreSQL data."""
        # Initialize with None instead of np.nan
        base_vector = {feat: None for feat in self.all_unique_features}
        previous_year = int(prediction_year - 1)  # Ensure it's a Python int
        
        # Get program history from PostgreSQL
        program_history = self._get_historical_data(program_id)
        
        # Get raw data for previous year
        raw_prev_year_data = self._get_raw_data(program_id)
        raw_prev_year_data = raw_prev_year_data[raw_prev_year_data['academicYear'] == previous_year]
                    
        # Fill using raw data for previous year
        if not raw_prev_year_data.empty:
            raw_data = self._convert_pandas_to_dict(raw_prev_year_data.iloc[0])
            for feat in base_vector:
                if feat.startswith("lag_") and not feat.endswith("_MA"):
                    source_col = feat[4:]
                    if source_col in raw_data and raw_data[source_col] is not None:
                        base_vector[feat] = convert_value(raw_data[source_col])

        # Get feature-engineered data for previous year
        prev_year_data = program_history[program_history['current_academicYear'] == previous_year]
        
        if not prev_year_data.empty:
            if not raw_prev_year_data.empty:
                # If we have raw data, use it to fill the base vector
                raw_prev_year_dict = self._convert_pandas_to_dict(raw_prev_year_data.iloc[0])
            prev_year_dict = self._convert_pandas_to_dict(prev_year_data.iloc[0])
            
            # Calculate MA features 
            for feat in base_vector:
                if feat.endswith("_MA") and base_vector[feat] is None:
                    base_feat = feat.replace("_MA", "").replace("lag_", "").replace("current_", "")
                    if base_feat in raw_prev_year_dict and feat in prev_year_dict:
                        current_val = raw_prev_year_dict[base_feat]
                        prev_ma = prev_year_dict[feat]
                        
                        if current_val is not None and prev_ma is not None:
                            n = len(program_history)
                            if n > 0:
                                new_ma = (float(prev_ma) * n + float(current_val)) / (n + 1)
                                base_vector[feat] = float(new_ma)
                        elif current_val is not None:
                            base_vector[feat] = convert_value(current_val)
                        elif prev_ma is not None:
                            base_vector[feat] = convert_value(prev_ma)
                        
            

            # # Then handle MA features
            # for feat in base_vector:
            #     if feat.endswith("_MA") and not (feat.startswith("lag_") or feat.startswith("current_")):
            #         base_feat = feat[:-3]
            #         if base_feat in raw_prev_year_dict and feat in prev_year_dict:
            #             n = len(program_history)
                        
            #             if n > 1:
            #                 prev_ma = prev_year_dict.get(feat)
            #                 if prev_ma is not None:
            #                     current_value = float(raw_prev_year_dict[base_feat])
            #                     prev_ma = float(prev_ma)
            #                     new_ma = (prev_ma * (n) + current_value) / n + 1
            #                     base_vector[feat] = float(new_ma)  
            #                 else:
            #                     base_vector[feat] = convert_value(prev_year_dict[feat])
            #             else:
            #                 base_vector[feat] = convert_value(raw_prev_year_dict[base_feat])
                            
            #     elif feat.startswith("lag_") and feat.endswith("_MA"):
            #         base_feat = feat[4:-3]
            #         if base_feat in raw_prev_year_dict and feat in prev_year_dict:
            #             n = len(program_history)
                        
            #             if n > 1:
            #                 prev_ma = prev_year_dict.get(feat)
            #                 if prev_ma is not None:
            #                     current_value = float(raw_prev_year_dict[base_feat])
            #                     prev_ma = float(prev_ma)
            #                     new_ma = (prev_ma * (n) + current_value) / n + 1
            #                     base_vector[feat] = float(new_ma)  
            #                 else:
            #                     base_vector[feat] = convert_value(prev_year_dict[feat])
            #             else:
            #                 base_vector[feat] = convert_value(raw_prev_year_dict[base_feat])
            #     elif feat.startswith("current_") and feat.endswith("_MA"):
            #         base_feat = feat[8:-3]
                    
            #         if base_feat in raw_prev_year_dict and feat in prev_year_dict:
            #             n = len(program_history)
                        
            #             if n > 1:
            #                 prev_ma = prev_year_dict.get(feat)
            #                 if prev_ma is not None:
            #                     current_value = float(raw_prev_year_dict[base_feat])
            #                     prev_ma = float(prev_ma)
            #                     new_ma = (prev_ma * (n) + current_value) / n + 1
            #                     base_vector[feat] = float(new_ma)  
            #                 else:
            #                     base_vector[feat] = convert_value(prev_year_dict[feat])
            #             else:
            #                 base_vector[feat] = convert_value(raw_prev_year_dict[base_feat])
                                
                # Fill other features directly from previous year data
                elif feat in prev_year_dict and prev_year_dict[feat] is not None and base_vector[feat] is None:
                    base_vector[feat] = convert_value(prev_year_dict[feat])

                # Fill economic indicators for 2025
                elif feat in ECONOMIC_INDICATORS_2025:
                    base_vector[feat] = ECONOMIC_INDICATORS_2025[feat]

        # Set index year
        if 'index_year' in base_vector:
            base_vector['index_year'] = float(prediction_year)
        
        # Final conversion to ensure everything is JSON-safe
        return self._convert_numpy_types(base_vector)

    def generate_and_save_to_file(self, program_ids, prediction_year, output_file="base_input_data.jsonl"):
        """Generate base input data for all programs and save to file"""
        successful = 0
        failed = []
        
        print(f"Generating base input data for {len(program_ids)} programs...")
        print(f"Saving to file: {output_file}")
        
        with open(output_file, 'w') as f:
            for i, program_id in enumerate(program_ids):
                try:
                    program_id = int(program_id)
                    base_input = self._get_base_input_vector(program_id, prediction_year)
                    now = datetime.utcnow().isoformat()
                    
                    # Verify JSON serialization works
                    try:
                        test_json = json.dumps(base_input, cls=NumpyEncoder)
                    except Exception as json_error:
                        print(f"JSON serialization failed for program {program_id}: {json_error}")
                        base_input = self._convert_numpy_types(base_input)
                    
                    row = {
                        "idOSYM": int(program_id),
                        "academicYear": int(prediction_year),
                        "created_at": now,
                        "base_input": json.dumps(base_input, cls=NumpyEncoder),
                        "last_updated": now
                    }
                    
                    # Write as newline-delimited JSON
                    f.write(json.dumps(row, cls=NumpyEncoder) + '\n')
                    successful += 1
                    
                    if successful % 1000 == 0:
                        print(f"Processed {successful} programs...")
                        
                except Exception as e:
                    failed.append((int(program_id), str(e)))
                    print(f"Failed for program {program_id}: {e}")
        
        print(f"Data generation complete! Processed {successful} programs successfully.")
        if failed:
            print(f"Failed to process {len(failed)} programs.")
        
        return successful, failed, output_file

    def load_file_to_bigquery(self, file_path):
        """Load the generated file to BigQuery"""
        print(f"Loading {file_path} to BigQuery table {self.table_ref}...")
        
        try:
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                write_disposition="WRITE_APPEND",
                # Don't specify schema - let it use the existing table schema
            )
            
            with open(file_path, "rb") as source_file:
                job = self.client.load_table_from_file(
                    source_file, self.table_ref, job_config=job_config
                )
            
            job.result()  # Wait for the load job to complete
            
            print(f"Successfully loaded data to BigQuery!")
            print(f"Loaded {job.output_rows} rows")
            
            return True
            
        except Exception as e:
            print(f"BigQuery loading failed: {e}")
            print(f"Data remains saved in {file_path} for manual loading")
            return False

    def load_existing_file(self, file_path):
        """Load an existing JSONL file to BigQuery without regenerating"""
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist!")
            return False
        
        stats = self.get_file_stats(file_path)
        if stats:
            print(f"Loading existing file: {file_path}")
            print(f"File contains {stats['records']} records ({stats['size_mb']} MB)")
        
        return self.load_file_to_bigquery(file_path)

    def get_file_stats(self, file_path):
        """Get statistics about an existing JSONL file"""
        if not os.path.exists(file_path):
            return None
        
        try:
            line_count = 0
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        line_count += 1
            
            file_size = os.path.getsize(file_path)
            return {
                'records': line_count,
                'size_mb': round(file_size / (1024 * 1024), 2)
            }
        except Exception as e:
            print(f"Error reading file stats: {e}")
            return None

    def process_all_programs(self, program_ids, prediction_year, keep_file=False, force_regenerate=False):
        """Complete workflow: generate data, save to file, load to BigQuery"""
        output_file = f"base_input_data_os_ranking_{prediction_year}.jsonl"
        
        # Check if file already exists
        if os.path.exists(output_file) and not force_regenerate:
            print(f"Found existing file: {output_file}")
            
            # Get file statistics
            stats = self.get_file_stats(output_file)
            if stats:
                print(f"File contains {stats['records']} records ({stats['size_mb']} MB)")
            
            print("Skipping data generation and proceeding to BigQuery loading...")
            print("(Use force_regenerate=True to regenerate the file)")
            successful = stats['records'] if stats else None
            failed = []
        else:
            if force_regenerate and os.path.exists(output_file):
                print(f"Force regenerate enabled. Overwriting existing file: {output_file}")
            
            # Step 1: Generate and save to file
            successful, failed, file_path = self.generate_and_save_to_file(
                program_ids, prediction_year, output_file
            )
        
        # Step 2: Load to BigQuery
        if os.path.exists(output_file):
            load_success = self.load_file_to_bigquery(output_file)
            
            # Step 3: Clean up file if requested and load was successful
            if load_success and not keep_file:
                try:
                    os.remove(output_file)
                    print(f"Cleaned up temporary file: {output_file}")
                except:
                    print(f"Could not remove temporary file: {output_file}")
            elif keep_file:
                print(f"Keeping data file: {output_file}")
        else:
            print(f"File {output_file} not found. Cannot proceed with loading.")
            load_success = False
        
        return successful, failed


if __name__ == "__main__":
    # Configuration
    MODEL_ASSETS_DIR = "./"
    RANKING_MODEL_OTHER_PROGRAMS_FILE = os.path.join(MODEL_ASSETS_DIR, 'last_unis_br_model.pkl')
    SERVICE_ACCOUNT_PATH = os.path.join(MODEL_ASSETS_DIR, "service-account-key.json")
    PROJECT_ID = "unioptima-461722"
    DATASET_ID = "university_db"
    TABLE_ID = "base_input_vectors_ranking_model"
    
    # Load the model to get features
    ranking_model_other = joblib.load(RANKING_MODEL_OTHER_PROGRAMS_FILE)
    features = ranking_model_other.feature_names_in_.tolist()
    
    # Create table (one-time setup)
    create_base_input_table(PROJECT_ID, DATASET_ID, TABLE_ID, SERVICE_ACCOUNT_PATH)
    
    # Initialize generator
    generator = BaseInputGenerator(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
        service_account_path=SERVICE_ACCOUNT_PATH,
        all_unique_features=features
    )
    
    # Get all unique program IDs from PostgreSQL
    train_df = generator._get_historical_data()
    
    # Get unique program IDs from historical data for current academic year = 2024
    train_df = train_df[train_df['current_academicYear'] == 2024]
    program_ids = train_df['idOSYM'].dropna().astype(int).unique()
    print(f"Found {len(program_ids)} unique programs in historical data")
    
    # Get current year (or set prediction year)
    prediction_year = 2025  # Predict for next year
    
    # Process all programs: generate file -> load to BigQuery
    # Set force_regenerate=True to recreate the file even if it exists
    successful, failed = generator.process_all_programs(
        program_ids, 
        prediction_year, 
        keep_file=True,  # Set to False to auto-delete the file after loading
        force_regenerate=False  # Set to True to regenerate even if file exists
    )
    
    # Alternative: To just load an existing file without the full workflow:
    # generator.load_existing_file("base_input_data_2025.jsonl")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE!")
    print(f"{'='*50}")
    if successful is not None:
        print(f"Successfully processed: {successful} programs")
    else:
        print("Used existing file - processing count unknown")
    print(f"Failed to process: {len(failed)} programs")
    
    if failed:
        print(f"\nFirst 10 failed programs:")
        for program_id, error in failed[:10]:
            print(f"Program {program_id}: {error}")
        
        # Save failed programs to CSV
        pd.DataFrame(failed, columns=['idOSYM', 'error']).to_csv(
            'failed_programs.csv', index=False)
        print(f"\nFull list of failed programs saved to 'failed_programs.csv'")
    
    print(f"\nBase input vectors generated for prediction year: {prediction_year}")
    print(f"Data available in BigQuery table: {PROJECT_ID}.{DATASET_ID}.{TABLE_ID}")