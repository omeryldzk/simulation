import os
import joblib
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import json
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    'occupied_slot_v1': {
        'table_id': 'base_input_vectors_v1',
        'model_dir': 'occupiedSlot',
        'model_file': 'occupied_slots_v1.pkl',
        'scaler_file': 'scaler_os_model_1.pkl'
    },
    'occupied_slot_v2': {
        'table_id': 'base_input_vectors_v2',
        'model_dir': 'occupiedSlot',
        'model_file': 'occupied_slots_v2.pkl',
        'scaler_file': 'scaler_os_model_2.pkl'
    },
    'occupied_slot_v3': {
        'table_id': 'base_input_vectors_v3',
        'model_dir': 'occupiedSlot',
        'model_file': 'occupied_slots_v3.pkl',
        'scaler_file': 'scaler_os_model_3.pkl'
    },
    'occupied_slot_v4': {
        'table_id': 'base_input_vectors_v4',
        'model_dir': 'occupiedSlot',
        'model_file': 'occupied_slots_v4.pkl',
        'scaler_file': 'scaler_os_model_4.pkl'
    },
    'base_ranking_top': {
        'table_id': 'base_input_vectors_ranking_model',
        'model_dir': 'baseRanking',
        'model_file': 'top_unis_br_model.pkl',
        'scaler_file': 'scaler_baseRanking.pkl',
        'ranking_threshold': 100000  # Use this model for rankings <= 100000
    },
    'base_ranking_last': {
        'table_id': 'base_input_vectors_ranking_model',
        'model_dir': 'baseRanking',
        'model_file': 'last_unis_br_model.pkl',
        'scaler_file': 'scaler_baseRanking.pkl',
        'ranking_threshold': float('inf')  # Use this model for rankings > 100000
    },
    'top_ranking': {
        'table_id': 'base_input_vectors_top_ranking_model',
        'model_dir': 'topRanking',
        'model_file': 'top_ranking_model.pkl',
        'scaler_file': 'scaler_topRanking.pkl'
    }
}

class RevenueOptimizer:
    def __init__(self, 
                 project_id: str,
                 dataset_id: str,
                 service_account_path: str,
                 models_dir: str = "models",
                 scalers_dir: str = "scalers"):
        """
        Initialize the revenue optimizer.
        
        Args:
            project_id: BigQuery project ID
            dataset_id: BigQuery dataset ID
            service_account_path: Path to service account JSON
            models_dir: Directory containing the model files
            scalers_dir: Directory containing the scaler files
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.models_dir = models_dir
        self.scalers_dir = scalers_dir
        
        # Initialize BigQuery client
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.client = bigquery.Client(credentials=credentials)
        
        # Load models and initialize model-specific tables
        self._load_models()
        
    def _load_models(self):
        """Load all required models and initialize their tables."""
        try:
            # Load occupied slot models and scalers
            self.occupied_slot_models = []
            self.occupied_slot_scalers = []
            self.occupied_slot_model_configs = []
            
            # Load each occupied slot model version and its scaler
            for model_key in ['occupied_slot_v1', 'occupied_slot_v2', 'occupied_slot_v3', 'occupied_slot_v4']:
                config = MODEL_CONFIGS[model_key]
                model_path = os.path.join(self.models_dir, config['model_dir'], config['model_file'])
                scaler_path = os.path.join(self.scalers_dir, config['scaler_file'])
                
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                self.occupied_slot_models.append(model)
                self.occupied_slot_scalers.append(scaler)
                self.occupied_slot_model_configs.append(config)
            
            # Load base ranking models and scaler
            self.base_ranking_models = {}
            self.base_ranking_scaler = joblib.load(os.path.join(self.scalers_dir, MODEL_CONFIGS['base_ranking_top']['scaler_file']))
            
            for model_key in ['base_ranking_top', 'base_ranking_last']:
                config = MODEL_CONFIGS[model_key]
                model_path = os.path.join(self.models_dir, config['model_dir'], config['model_file'])
                self.base_ranking_models[model_key] = joblib.load(model_path)
            
            # Load top ranking model and scaler
            top_ranking_path = os.path.join(self.models_dir, MODEL_CONFIGS['top_ranking']['model_dir'], MODEL_CONFIGS['top_ranking']['model_file'])
            top_ranking_scaler_path = os.path.join(self.scalers_dir, MODEL_CONFIGS['top_ranking']['scaler_file'])
            self.top_ranking_model = joblib.load(top_ranking_path)
            self.top_ranking_scaler = joblib.load(top_ranking_scaler_path)
            
            logger.info(f"Loaded {len(self.occupied_slot_models)} occupied slot models and scalers")
            logger.info(f"Loaded {len(self.base_ranking_models)} base ranking models and scaler")
            logger.info("Loaded top ranking model and scaler")
            
        except Exception as e:
            logger.error(f"Error loading models and scalers: {e}")
            raise
    
    def get_base_input_vectors(self, program_ids: List[int], prediction_year: int, model_type: str) -> pd.DataFrame:
        """
        Fetch base input vectors from BigQuery for specified programs and model type.
        
        Args:
            program_ids: List of program IDs to fetch vectors for
            prediction_year: Year to predict for
            model_type: Type of model (e.g., 'occupied_slot_v1', 'base_ranking_top', 'top_ranking')
            
        Returns:
            DataFrame containing base input vectors
        """
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Invalid model type: {model_type}")
            
        table_id = MODEL_CONFIGS[model_type]['table_id']
        query = f"""
        SELECT idOSYM, base_input
        FROM `{self.project_id}.{self.dataset_id}.{table_id}`
        WHERE idOSYM IN UNNEST(@program_ids)
        AND academicYear = @prediction_year
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("program_ids", "INT64", program_ids),
                bigquery.ScalarQueryParameter("prediction_year", "INT64", prediction_year)
            ]
        )
        
        try:
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            # Convert results to DataFrame
            data = []
            program_data = {}
            for row in results:
                base_input = json.loads(row.base_input)
                data.append(base_input)
                program_data[row.idOSYM] = base_input
                logger.info(f"Base Input Shape for Program ID {row.idOSYM} ({model_type}): {len(base_input)} features")
            
            if not data:
                logger.warning(f"No base input vectors found for {model_type} and programs {program_ids}")
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            logger.info(f"Total Base Input Shape for {model_type}: {len(df.columns)} features, {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching base input vectors for {model_type}: {e}")
            raise
    
    def predict_occupied_slots(self, program_id: int, prediction_year: int, quota: int = None, fee: float = None) -> float:
        """
        Predict number of occupied slots using ensemble of occupied slot models.
        
        Args:
            program_id: Program ID to predict for
            prediction_year: Year to predict for
            quota: Optional quota to override in base input
            fee: Optional fee to override in base input
            
        Returns:
            Average predicted occupied slots from all models
        """
        predictions = []
        
        for i, (model, scaler, config) in enumerate(zip(self.occupied_slot_models, self.occupied_slot_scalers, self.occupied_slot_model_configs)):
            # Get base input vectors for this specific model version
            base_input = self.get_base_input_vectors([program_id], prediction_year, f'occupied_slot_v{i+1}')
            
            if base_input.empty:
                logger.error(f"No base input found for program {program_id} with model v{i+1}")
                continue
            
            # Make a copy to avoid modifying original data
            input_df = base_input.copy()
            
            # Override quota and fee if provided
            if quota is not None and 'quota' in input_df.columns:
                input_df['quota'] = quota
            if fee is not None and 'fee' in input_df.columns:
                input_df['fee'] = fee
            
            # Scale the input features
            scale_features = scaler.feature_names_in_.tolist()
            rest_features = [col for col in input_df.columns if col not in scale_features]
            
            logger.info(f"Occupied Slot Model v{i+1}:")
            logger.info(f"  Base input shape: {input_df.shape}")
            logger.info(f"  Scale features: {len(scale_features)}")
            logger.info(f"  Rest features: {len(rest_features)}")
            logger.info(f"  Model expected features: {model.n_features_in_}")
            
            # Prepare input features
            scale_input = input_df[scale_features]
            scaled_input = scaler.transform(scale_input)
            
            if rest_features:
                rest_input = input_df[rest_features].values
                final_input = np.hstack((scaled_input, rest_input))
            else:
                final_input = scaled_input
            
            logger.info(f"  Final input shape: {final_input.shape}")
            
            # Make prediction
            pred = model.predict(final_input)[0]
            predictions.append(pred)
            logger.info(f"Occupied Slot Model v{i+1} completed")
            logger.info(f"  Prediction: {pred}")
        
        if not predictions:
            raise ValueError(f"No predictions could be made for program {program_id}")
        
        # Return average prediction
        avg_prediction = np.mean(predictions)
        logger.info(f"Average occupied slots prediction: {avg_prediction}")
        return avg_prediction
    
    def predict_base_ranking(self, program_id: int, prediction_year: int, quota: int = None, fee: float = None) -> float:
        """
        Predict base ranking using base ranking models.
        
        Args:
            program_id: Program ID to predict for
            prediction_year: Year to predict for
            quota: Optional quota to override in base input
            fee: Optional fee to override in base input
            
        Returns:
            Predicted base ranking
        """
        # Get base input vectors for base ranking model
        base_input = self.get_base_input_vectors([program_id], prediction_year, 'base_ranking_top')
        
        if base_input.empty:
            raise ValueError(f"No base input vectors found for base ranking model and program {program_id}")
        
        # Make a copy to avoid modifying original data
        input_df = base_input.copy()
        
        # Override quota and fee if provided
        if quota is not None and 'quota' in input_df.columns:
            input_df['quota'] = quota
        if fee is not None and 'fee' in input_df.columns:
            input_df['fee'] = fee
        
        # Determine which model to use based on historical ranking
        lag_base_ranking = input_df['lag_baseRanking'].iloc[0] if 'lag_baseRanking' in input_df.columns else 0
        
        # Choose appropriate model
        if lag_base_ranking <= MODEL_CONFIGS['base_ranking_top']['ranking_threshold']:
            selected_model = self.base_ranking_models['base_ranking_top']
            model_name = 'base_ranking_top'
        else:
            selected_model = self.base_ranking_models['base_ranking_last']
            model_name = 'base_ranking_last'
        
        # Scale the input features
        scale_features = self.base_ranking_scaler.feature_names_in_.tolist()
        rest_features = [col for col in input_df.columns if col not in scale_features]
        
        logger.info(f"Base Ranking Model ({model_name}):")
        logger.info(f"  Lag base ranking: {lag_base_ranking}")
        logger.info(f"  Base input shape: {input_df.shape}")
        logger.info(f"  Scale features: {len(scale_features)}")
        logger.info(f"  Rest features: {len(rest_features)}")
        logger.info(f"  Model expected features: {selected_model.n_features_in_}")
        
        # Prepare input features
        scale_input = input_df[scale_features]
        scaled_input = self.base_ranking_scaler.transform(scale_input)
        
        if rest_features:
            rest_input = input_df[rest_features].values
            final_input = np.hstack((scaled_input, rest_input))
        else:
            final_input = scaled_input
        
        logger.info(f"  Final input shape: {final_input.shape}")
        
        # Make prediction
        prediction = selected_model.predict(final_input)[0]
        logger.info(f"  Base ranking prediction: {prediction}")
        
        return prediction
    
    def predict_top_ranking(self, program_id: int, prediction_year: int, quota: int = None, fee: float = None) -> float:
        """
        Predict top ranking using top ranking model.
        
        Args:
            program_id: Program ID to predict for
            prediction_year: Year to predict for
            quota: Optional quota to override in base input
            fee: Optional fee to override in base input
            
        Returns:
            Predicted top ranking
        """
        # Get base input vectors for top ranking model
        base_input = self.get_base_input_vectors([program_id], prediction_year, 'top_ranking')
        
        if base_input.empty:
            raise ValueError(f"No base input vectors found for top ranking model and program {program_id}")
        
        # Make a copy to avoid modifying original data
        input_df = base_input.copy()
        
        # Override quota and fee if provided
        if quota is not None and 'quota' in input_df.columns:
            input_df['quota'] = quota
        if fee is not None and 'fee' in input_df.columns:
            input_df['fee'] = fee
        
        # Scale the input features
        scale_features = self.top_ranking_scaler.feature_names_in_.tolist()
        rest_features = [col for col in input_df.columns if col not in scale_features]
        
        logger.info(f"Top Ranking Model:")
        logger.info(f"  Base input shape: {input_df.shape}")
        logger.info(f"  Scale features: {len(scale_features)}")
        logger.info(f"  Rest features: {len(rest_features)}")
        logger.info(f"  Model expected features: {self.top_ranking_model.n_features_in_}")
        
        # Prepare input features
        scale_input = input_df[scale_features]
        scaled_input = self.top_ranking_scaler.transform(scale_input)
        
        if rest_features:
            rest_input = input_df[rest_features].values
            final_input = np.hstack((scaled_input, rest_input))
        else:
            final_input = scaled_input
        
        logger.info(f"  Final input shape: {final_input.shape}")
        
        # Make prediction
        prediction = self.top_ranking_model.predict(final_input)[0]
        logger.info(f"  Top ranking prediction: {prediction}")
        
        return prediction
    
    def optimize_revenue(self, 
                        program_id: int,
                        current_quota: int,
                        current_fee: float,
                        min_ranking_threshold: float = 0.9,
                        max_quota_increase: float = 0.5,
                        fee_step: float = 1000.0,
                        quota_step: int = 5) -> Dict:
        """
        Optimize revenue by finding optimal quota and fee combination.
        
        Args:
            program_id: Program ID to optimize
            current_quota: Current quota of the program
            current_fee: Current fee of the program
            min_ranking_threshold: Minimum acceptable ranking (as percentage of current ranking)
            max_quota_increase: Maximum allowed quota increase as percentage
            fee_step: Step size for fee optimization
            quota_step: Step size for quota optimization
            
        Returns:
            Dictionary containing optimization results
        """
        prediction_year = 2025
        
        # Get current predictions
        logger.info(f"Getting current predictions for program {program_id}")
        
        current_occupied = self.predict_occupied_slots(program_id, prediction_year, current_quota, current_fee)
        current_base_rank = self.predict_base_ranking(program_id, prediction_year, current_quota, current_fee)
        current_top_rank = self.predict_top_ranking(program_id, prediction_year, current_quota, current_fee)
        
        logger.info(f"Current occupied slots: {current_occupied}")
        logger.info(f"Current base ranking: {current_base_rank}")
        logger.info(f"Current top ranking: {current_top_rank}")
        
        # Calculate current revenue
        current_revenue = current_occupied * current_fee
        
        # Initialize optimization variables
        best_revenue = current_revenue
        best_quota = current_quota
        best_fee = current_fee
        best_occupied = current_occupied
        best_base_rank = current_base_rank
        
        max_quota = int(current_quota * (1 + max_quota_increase))
        max_fee = current_fee * 2
        
        logger.info(f"Starting optimization:")
        logger.info(f"  Quota range: {current_quota} to {max_quota} (step: {quota_step})")
        logger.info(f"  Fee range: {current_fee} to {max_fee} (step: {fee_step})")
        
        # # Grid search for optimal combination
        # iteration = 0
        # for quota in range(current_quota, max_quota + 1, quota_step):
        #     for fee in np.arange(current_fee, max_fee, fee_step):
        #         iteration += 1
                
        #         try:
        #             # Get predictions for this combination
        #             occupied = self.predict_occupied_slots(program_id, prediction_year, quota, fee)
        #             base_rank = self.predict_base_ranking(program_id, prediction_year, quota, fee)
                    
        #             # Calculate revenue
        #             revenue = occupied * fee
                    
        #             # Check if ranking constraint is satisfied
        #             if base_rank <= current_base_rank * min_ranking_threshold:
        #                 if revenue > best_revenue:
        #                     best_revenue = revenue
        #                     best_quota = quota
        #                     best_fee = fee
        #                     best_occupied = occupied
        #                     best_base_rank = base_rank
                            
        #                     logger.info(f"New best found (iteration {iteration}): quota={quota}, fee={fee:.0f}, revenue={revenue:.0f}, rank={base_rank:.0f}")
                
        #         except Exception as e:
        #             logger.warning(f"Error in iteration {iteration} (quota={quota}, fee={fee}): {e}")
        #             continue
        
        # logger.info(f"Optimization completed after {iteration} iterations")
        
        return {
            'program_id': program_id,
            'current_quota': current_quota,
            'current_fee': current_fee,
            'current_revenue': current_revenue,
            'current_occupied': current_occupied,
            'current_base_ranking': current_base_rank,
            'current_top_ranking': current_top_rank,
            'optimal_quota': best_quota,
            'optimal_fee': best_fee,
            'optimal_revenue': best_revenue,
            'optimal_occupied': best_occupied,
            'optimal_base_ranking': best_base_rank,
            'revenue_increase': best_revenue - current_revenue,
            'revenue_increase_percentage': ((best_revenue - current_revenue) / current_revenue) * 100 if current_revenue > 0 else 0
        }

def main():
    # Configuration
    PROJECT_ID = "unioptima-461722"
    DATASET_ID = "university_db"
    SERVICE_ACCOUNT_PATH = "service-account-key.json"
    
    # Initialize optimizer
    optimizer = RevenueOptimizer(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        service_account_path=SERVICE_ACCOUNT_PATH
    )
    
    # Example usage
    program_id = 203852027  # Replace with actual program ID
    current_quota = 10
    current_fee = 50000
    min_ranking_threshold = 0.9  # Allow 10% ranking decrease
    
    try:
        results = optimizer.optimize_revenue(
            program_id=program_id,
            current_quota=current_quota,
            current_fee=current_fee,
            min_ranking_threshold=min_ranking_threshold
        )
        
        print("\n" + "="*80)
        print("REVENUE OPTIMIZATION RESULTS")
        print("="*80)
        print(f"Program ID: {results['program_id']}")
        print(f"\nCURRENT STATE:")
        print(f"  Quota: {results['current_quota']}")
        print(f"  Fee: {results['current_fee']:,.2f} TL")
        print(f"  Revenue: {results['current_revenue']:,.2f} TL")
        print(f"  Occupied Slots: {results['current_occupied']:.1f}")
        print(f"  Base Ranking: {results['current_base_ranking']:.0f}")
        print(f"  Top Ranking: {results['current_top_ranking']:.0f}")
        
        print(f"\nOPTIMAL STATE:")
        print(f"  Quota: {results['optimal_quota']}")
        print(f"  Fee: {results['optimal_fee']:,.2f} TL")
        print(f"  Revenue: {results['optimal_revenue']:,.2f} TL")
        print(f"  Occupied Slots: {results['optimal_occupied']:.1f}")
        print(f"  Base Ranking: {results['optimal_base_ranking']:.0f}")
        
        print(f"\nIMPROVEMENT:")
        print(f"  Revenue Increase: {results['revenue_increase']:,.2f} TL")
        print(f"  Revenue Increase %: {results['revenue_increase_percentage']:.2f}%")
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}")

if __name__ == "__main__":
    main()