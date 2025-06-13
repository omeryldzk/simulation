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
            all_unique_features: List of features specific to this model
        """
        self.engine = create_engine(DATABASE_URL)
        self.all_unique_features = all_unique_features
        if service_account_path:
            self.client = get_bigquery_client(service_account_path)
        else:
            self.client = bigquery.Client(project=project_id)
            
        self.table_ref = f"{project_id}.{dataset_id}.{table_id}"
        
        print(f"✅ BaseInputGenerator initialized with {len(self.all_unique_features)} features")
        print(f"   First 5 features: {self.all_unique_features[:5]}")
    
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
        table_name = "encoded_data_os_avg"
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
        # Initialize with None instead of np.nan - ONLY for features this model needs
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
                                
                # Fill other features directly from previous year data
                elif feat in prev_year_dict and prev_year_dict[feat] is not None:
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
        print(f"Each record will have {len(self.all_unique_features)} features")
        
        with open(output_file, 'w') as f:
            for i, program_id in enumerate(program_ids):
                try:
                    program_id = int(program_id)
                    base_input = self._get_base_input_vector(program_id, prediction_year)
                    now = datetime.utcnow().isoformat()
                    
                    # Verify the feature count matches expectation
                    if len(base_input) != len(self.all_unique_features):
                        print(f"⚠️  WARNING: Feature count mismatch for program {program_id}")
                        print(f"   Expected: {len(self.all_unique_features)}, Got: {len(base_input)}")
                    
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

    def process_all_programs(self, program_ids, prediction_year, model_suffix="", keep_file=False, force_regenerate=False):
        """Complete workflow with model-specific file naming and feature verification"""
        # FIXED: Make the output file name model-specific
        if model_suffix:
            output_file = f"base_input_data_os_ranking_{prediction_year}_{model_suffix}.jsonl"
        else:
            output_file = f"base_input_data_os_ranking_{prediction_year}.jsonl"
        
        print(f"Using output file: {output_file}")
        print(f"✅ Feature count for this model: {len(self.all_unique_features)}")
        
        # Check if file already exists
        if os.path.exists(output_file) and not force_regenerate:
            print(f"Found existing file: {output_file}")
            
            # Get file statistics and verify feature count
            stats = self.get_file_stats(output_file)
            if stats:
                print(f"File contains {stats['records']} records ({stats['size_mb']} MB)")
                
                # Verify the feature count in the existing file
                try:
                    with open(output_file, 'r') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            record = json.loads(first_line)
                            base_input = json.loads(record['base_input'])
                            file_feature_count = len(base_input)
                            
                            print(f"✅ Existing file has {file_feature_count} features")
                            
                            if file_feature_count != len(self.all_unique_features):
                                print(f"⚠️  WARNING: Feature count mismatch!")
                                print(f"   Model expects: {len(self.all_unique_features)} features")
                                print(f"   File contains: {file_feature_count} features")
                                print(f"   Force regenerating file...")
                                force_regenerate = True
                except Exception as e:
                    print(f"❌ Error reading existing file: {e}")
                    print("Force regenerating file...")
                    force_regenerate = True
            
            if not force_regenerate:
                print("Skipping data generation and proceeding to BigQuery loading...")
                successful = stats['records'] if stats else None
                failed = []
        
        if force_regenerate or not os.path.exists(output_file):
            if force_regenerate and os.path.exists(output_file):
                print(f"Force regenerate enabled. Overwriting existing file: {output_file}")
            
            # Step 1: Generate and save to file
            successful, failed, file_path = self.generate_and_save_to_file(
                program_ids, prediction_year, output_file
            )
            
            # Verify the generated file has correct feature count
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            record = json.loads(first_line)
                            base_input = json.loads(record['base_input'])
                            generated_feature_count = len(base_input)
                            
                            print(f"✅ Generated file has {generated_feature_count} features")
                            
                            if generated_feature_count != len(self.all_unique_features):
                                print(f"❌ ERROR: Feature count mismatch after generation!")
                                print(f"   Expected: {len(self.all_unique_features)}")
                                print(f"   Generated: {generated_feature_count}")
                except Exception as e:
                    print(f"❌ Error verifying generated file: {e}")
        
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

def process_model(model_name, model_config, program_ids, prediction_year=2025):
    """Process a single model and generate base input data with enhanced debugging"""
    print(f"\n{'='*60}")
    print(f"PROCESSING MODEL: {model_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Load the model to get features
        model_path = os.path.join(MODEL_ASSETS_DIR, model_config['model_file'])
        print(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        ranking_model = joblib.load(model_path)
        features = ranking_model.feature_names_in_.tolist()
        print(f"✅ Found {len(features)} features for model {model_name}")
        print(f"   First 10 features: {features[:10]}")
        print(f"   Last 5 features: {features[-5:]}")
        
        # Create table ID specific to this model
        table_id = f"{BASE_TABLE_ID}_{model_config['table_suffix']}"
        
        # Create table (one-time setup for this model)
        print(f"Creating/checking table: {PROJECT_ID}.{DATASET_ID}.{table_id}")
        create_base_input_table(PROJECT_ID, DATASET_ID, table_id, SERVICE_ACCOUNT_PATH)
        
        # Initialize generator for this model with model-specific features
        generator = BaseInputGenerator(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            table_id=table_id,
            service_account_path=SERVICE_ACCOUNT_PATH,
            all_unique_features=features  # Each model gets its own features
        )
        
        # Verify the generator has the correct features
        print(f"✅ Generator initialized with {len(generator.all_unique_features)} features")
        
        # Process all programs for this model with model-specific output file
        print(f"Processing {len(program_ids)} programs for model {model_name}")
        successful, failed = generator.process_all_programs(
            program_ids, 
            prediction_year,
            model_suffix=model_config['table_suffix'],  # FIXED: Make file names unique
            keep_file=True,
            force_regenerate=False
        )
        
        # Print summary for this model
        print(f"\n{'-'*40}")
        print(f"MODEL {model_name.upper()} SUMMARY:")
        print(f"{'-'*40}")
        if successful is not None:
            print(f"Successfully processed: {successful} programs")
        else:
            print("Used existing file - processing count unknown")
        print(f"Failed to process: {len(failed)} programs")
        print(f"Features in this model: {len(features)}")
        print(f"Data available in BigQuery table: {PROJECT_ID}.{DATASET_ID}.{table_id}")
        
        return {
            'model_name': model_name,
            'successful': successful,
            'failed': failed,
            'table_id': table_id,
            'features_count': len(features)
        }
        
    except Exception as e:
        print(f"❌ ERROR processing model {model_name}: {str(e)}")
        return {
            'model_name': model_name,
            'successful': 0,
            'failed': [(0, f"Model loading error: {str(e)}")],
            'table_id': None,
            'features_count': 0
        }

def main():
    """Main function to process all models with better debugging"""
    print("Starting multi-model base input generation...")
    
    # First, debug the models to understand feature differences
    print(f"\n{'='*60}")
    print("DEBUGGING MODEL FEATURES")
    print(f"{'='*60}")
    
    model_features = {}
    all_model_features = {}
    
    for model_name, model_config in MODELS_CONFIG.items():
        try:
            model_path = os.path.join(MODEL_ASSETS_DIR, model_config['model_file'])
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                features = model.feature_names_in_.tolist()
                model_features[model_name] = len(features)
                all_model_features[model_name] = set(features)
                print(f"Model {model_name}: {len(features)} features")
            else:
                print(f"❌ Model file not found: {model_path}")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
    
    # Check if all models have the same number of features
    if len(set(model_features.values())) == 1:
        print(f"\n⚠️  WARNING: All models have the same number of features ({list(model_features.values())[0]})")
        
        # Check if they actually have the same features
        if len(all_model_features) > 1:
            model_names = list(all_model_features.keys())
            base_features = all_model_features[model_names[0]]
            all_identical = True
            
            for model_name in model_names[1:]:
                if all_model_features[model_name] != base_features:
                    all_identical = False
                    break
            
            if all_identical:
                print("   All models have IDENTICAL feature sets!")
                print("   This explains why base input vectors have the same shape.")
            else:
                print("   Models have DIFFERENT feature sets despite same count!")
    else:
        print(f"\n✅ Models have different numbers of features (as expected)")
        for model_name, count in model_features.items():
            print(f"   {model_name}: {count} features")
    
    # Get historical data
    print(f"\n{'='*60}")
    print("GETTING HISTORICAL DATA")
    print(f"{'='*60}")
    
    # Use any available model to get historical data
    available_model = None
    for model_name, model_config in MODELS_CONFIG.items():
        model_path = os.path.join(MODEL_ASSETS_DIR, model_config['model_file'])
        if os.path.exists(model_path):
            available_model = (model_name, model_config)
            break
    
    if not available_model:
        print("❌ No available models found!")
        return
    
    model_name, model_config = available_model
    model_path = os.path.join(MODEL_ASSETS_DIR, model_config['model_file'])
    temp_model = joblib.load(model_path)
    temp_features = temp_model.feature_names_in_.tolist()
    
    # Create temporary generator just to get historical data
    temp_generator = BaseInputGenerator(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id="temp",
        service_account_path=SERVICE_ACCOUNT_PATH,
        all_unique_features=temp_features
    )
    
    # Get program IDs
    train_df = temp_generator._get_historical_data()
    train_df = train_df[train_df['current_academicYear'] == 2024]
    program_ids = train_df['idOSYM'].dropna().astype(int).unique()
    print(f"Found {len(program_ids)} unique programs in historical data")
    
    # Set prediction year
    prediction_year = 2025
    
    # Process each model
    results = []
    all_failed_programs = []
    
    for model_name, model_config in MODELS_CONFIG.items():
        result = process_model(model_name, model_config, program_ids, prediction_year)
        results.append(result)
        
        # Collect failed programs with model info
        if result['failed']:
            for program_id, error in result['failed']:
                all_failed_programs.append({
                    'model': model_name,
                    'idOSYM': program_id,
                    'error': error
                })
    
    # Print overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL PROCESSING COMPLETE!")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\nModel {result['model_name'].upper()}:")
        print(f"  - Features: {result['features_count']}")
        if result['successful'] is not None:
            print(f"  - Successful: {result['successful']} programs")
        else:
            print(f"  - Used existing file")
        print(f"  - Failed: {len(result['failed'])} programs")
        if result['table_id']:
            print(f"  - Table: {PROJECT_ID}.{DATASET_ID}.{result['table_id']}")
    
    # Save failed programs
    if all_failed_programs:
        failed_df = pd.DataFrame(all_failed_programs)
        failed_df.to_csv('failed_programs_all_models.csv', index=False)
        print(f"\nFailed programs saved to 'failed_programs_all_models.csv'")
        
        failure_summary = failed_df.groupby('model').size()
        print(f"\nFailure summary by model:")
        for model, count in failure_summary.items():
            print(f"  - {model}: {count} failures")
    
    print(f"\nBase input vectors generated for prediction year: {prediction_year}")
    print(f"Data available in BigQuery tables: {PROJECT_ID}.{DATASET_ID}.{BASE_TABLE_ID}_{{v1,v2,v3,v4}}")

if __name__ == "__main__":
    # Configuration for multiple models
    MODELS_CONFIG = {
        'v1': {
            'model_file': 'occupied_slots_v1.pkl',
            'table_suffix': 'v1'
        },
        'v2': {
            'model_file': 'occupied_slots_v2.pkl',
            'table_suffix': 'v2'
        },
        'v3': {
            'model_file': 'occupied_slots_v3.pkl',
            'table_suffix': 'v3'
        },
        'v4': {
            'model_file': 'occupied_slots_v4.pkl',
            'table_suffix': 'v4'
        }
    }
    
    MODEL_ASSETS_DIR = "./"
    SERVICE_ACCOUNT_PATH = os.path.join(MODEL_ASSETS_DIR, "service-account-key.json")
    PROJECT_ID = "unioptima-461722"
    DATASET_ID = "university_db"
    BASE_TABLE_ID = "base_input_vectors"
    main()