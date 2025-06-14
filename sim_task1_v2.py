import os
import joblib
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import json
from typing import Dict, List, Tuple, Optional
import logging
import math

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
        'ranking_threshold': 100000
    },
    'base_ranking_last': {
        'table_id': 'base_input_vectors_ranking_model',
        'model_dir': 'baseRanking',
        'model_file': 'last_unis_br_model.pkl',
        'scaler_file': 'scaler_baseRanking.pkl',
        'ranking_threshold': float('inf')
    },
    'top_ranking': {
        'table_id': 'base_input_vectors_top_ranking_model',
        'model_dir': 'topRanking',
        'model_file': 'top_ranking_model.pkl',
        'scaler_file': 'scaler_topRanking.pkl'
    }
}

def compute_ranking_stability(current_rank):
    # Zone 1: Slow change (0–10,000) → stability ≈ 0.2
    # Zone 2: Moderate change (10,000–100,000) → stability ≈ 0.5
    # Zone 3: Fast change (>100,000) → stability ≈ 1.0

    # Sigmoid for transition between Zone 1 and Zone 2 (centered at 10,000)
    transition1 = 1 / (1 + math.exp(-(current_rank - 10000) / 3000))  # 3000 = smoothness

    # Sigmoid for transition between Zone 2 and Zone 3 (centered at 100,000)
    transition2 = 1 / (1 + math.exp(-(current_rank - 100000) / 20000))  # 20000 = smoothness

    # Combined effect (weights transitions)
    stability = 0.2 + (0.3 * transition1) + (0.5 * transition2)
    return min(stability, 1.0)  # Ensure ≤ 1.0


class DynamicRevenueOptimizer:
    def __init__(self, 
                 project_id: str,
                 dataset_id: str,
                 service_account_path: str,
                 models_dir: str = "models",
                 scalers_dir: str = "scalers"):
        """
        Initialize the dynamic revenue optimizer.
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
        """Fetch base input vectors from BigQuery for specified programs and model type."""
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
            
            data = []
            for row in results:
                base_input = json.loads(row.base_input)
                data.append(base_input)
            
            if not data:
                logger.warning(f"No base input vectors found for {model_type} and programs {program_ids}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching base input vectors for {model_type}: {e}")
            raise

    def calculate_dynamic_weights(self, base_ranking: float, iteration: int, total_iterations: int) -> List[float]:
        """
        Calculate dynamic weights for occupied slot models based on base ranking and iteration.
        
        Args:
            base_ranking: Current base ranking (lower is better)
            iteration: Current iteration number
            total_iterations: Total number of iterations
            
        Returns:
            List of weights for [v1, v2, v3, v4] models
        """
        # Normalize base ranking to 0-1 scale (assuming max ranking around 500000)
        normalized_ranking = min(base_ranking / 500000.0, 1.0)
        
        # Progress factor (0 to 1)
        progress = iteration / total_iterations
        
        # Base weights
        base_weights = [0.25, 0.25, 0.25, 0.25]
        
        # If base ranking is near 0 (good ranking), increase v1 weight
        ranking_factor = 1 - normalized_ranking  # Higher when ranking is better
        
        # Dynamic adjustments
        # v1: Higher weight for better rankings (programs near 0)
        v1_boost = ranking_factor * 0.3
        
        # v2: Stable weight
        v2_adjustment = 0.0
        
        # v3: Higher weight as iterations progress (quota effect becomes more important)
        # Also higher for worse rankings (more sensitive to quota changes)
        v3_boost = progress * 0.2 + normalized_ranking * 0.15
        
        # v4: Complementary weight
        v4_adjustment = -(v1_boost + v3_boost) / 2
        
        # Apply adjustments
        weights = [
            base_weights[0] + v1_boost,
            base_weights[1] + v2_adjustment,
            base_weights[2] + v3_boost,
            base_weights[3] + v4_adjustment
        ]
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Ensure no negative weights
        weights = [max(0.05, w) for w in weights]  # Minimum 5% weight
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return weights

    def predict_occupied_slots_individual(self, program_id: int, prediction_year: int, quota: int = None, fee: float = None) -> List[float]:
        """
        Predict occupied slots using individual models (returns list of predictions).
        
        Returns:
            List of predictions from [v1, v2, v3, v4] models
        """
        predictions = []
        
        for i, (model, scaler, config) in enumerate(zip(self.occupied_slot_models, self.occupied_slot_scalers, self.occupied_slot_model_configs)):
            # Get base input vectors for this specific model version
            base_input = self.get_base_input_vectors([program_id], prediction_year, f'occupied_slot_v{i+1}')
            quota = base_input['lag_occupiedSlots'].iloc[0] if 'lag_occupiedSlots' in base_input.columns else 0

            if base_input.empty:
                logger.error(f"No base input found for program {program_id} with model v{i+1}")
                predictions.append(0)
                continue
            
            # Make a copy to avoid modifying original data
            input_df = base_input.copy()
            
            # Override quota if provided
            if quota is not None and 'current_quota' in input_df.columns:
                input_df['current_quota'] = quota
            if fee is not None and 'current_fee' in input_df.columns:
                input_df['current_fee'] = fee
            
            # Scale the input features
            scale_features = scaler.feature_names_in_.tolist()
            rest_features = [col for col in input_df.columns if col not in scale_features]
            
            # Prepare input features
            scale_input = input_df[scale_features]
            scaled_input = scaler.transform(scale_input)
            
            if rest_features:
                rest_input = input_df[rest_features].values
                final_input = np.hstack((scaled_input, rest_input))
            else:
                final_input = scaled_input
            
            # Make prediction
            pred = model.predict(final_input)[0]
            predictions.append(pred)
        
        return predictions

    def predict_base_ranking_simple(self, program_id: int, prediction_year: int, fee: float = None) -> float:
        """Simplified base ranking prediction."""
        base_input = self.get_base_input_vectors([program_id], prediction_year, 'base_ranking_top')
        quota = base_input['lag_occupiedSlots'].iloc[0] if 'lag_occupiedSlots' in base_input.columns else 0
        
        if base_input.empty:
            raise ValueError(f"No base input vectors found for base ranking model and program {program_id}")
        
        input_df = base_input.copy()

        if quota is not None and 'current_quota' in input_df.columns:
            input_df['current_quota'] = quota
        if fee is not None and 'current_fee' in input_df.columns:
            input_df['current_fee'] = fee
        
        lag_base_ranking = input_df['lag_baseRanking'].iloc[0] if 'lag_baseRanking' in input_df.columns else 0
        
        if lag_base_ranking <= MODEL_CONFIGS['base_ranking_top']['ranking_threshold']:
            selected_model = self.base_ranking_models['base_ranking_top']
        else:
            selected_model = self.base_ranking_models['base_ranking_last']
        
        scale_features = self.base_ranking_scaler.feature_names_in_.tolist()
        rest_features = [col for col in input_df.columns if col not in scale_features]
        
        scale_input = input_df[scale_features]
        scaled_input = self.base_ranking_scaler.transform(scale_input)
        
        if rest_features:
            rest_input = input_df[rest_features].values
            final_input = np.hstack((scaled_input, rest_input))
        else:
            final_input = scaled_input
        
        prediction = selected_model.predict(final_input)[0]
        return prediction

    def predict_top_ranking_simple(self, program_id: int, prediction_year: int,fee: float = None) -> float:
        """Simplified top ranking prediction."""
        base_input = self.get_base_input_vectors([program_id], prediction_year, 'top_ranking')
        quota = base_input['lag_occupiedSlots'].iloc[0] if 'lag_occupiedSlots' in base_input.columns else 0

        if base_input.empty:
            raise ValueError(f"No base input vectors found for top ranking model and program {program_id}")
        
        input_df = base_input.copy()

        if quota is not None and 'current_quota' in input_df.columns:
            input_df['current_quota'] = quota
        if fee is not None and 'current_fee' in input_df.columns:
            input_df['current_fee'] = fee
        
        scale_features = self.top_ranking_scaler.feature_names_in_.tolist()
        rest_features = [col for col in input_df.columns if col not in scale_features]
        
        scale_input = input_df[scale_features]
        scaled_input = self.top_ranking_scaler.transform(scale_input)
        
        if rest_features:
            rest_input = input_df[rest_features].values
            final_input = np.hstack((scaled_input, rest_input))
        else:
            final_input = scaled_input
        
        prediction = self.top_ranking_model.predict(final_input)[0]
        return prediction

    def dynamic_optimize_revenue(self, 
                                program_id: int,
                                current_fee: float,
                                min_quota: int,
                                max_quota: int,
                                threshold: float = 100000,
                                prediction_year: int = 2025) -> Dict:
        """
        Dynamic optimization with adaptive weighting and logarithmic ranking changes.
        
        Args:
            program_id: Program ID to optimize
            current_fee: Current fee
            min_quota: Minimum quota to test
            max_quota: Maximum quota to test
            prediction_year: Year to predict for
            
        Returns:
            Dictionary containing optimization results
        """
        
        logger.info(f"Starting dynamic optimization for program {program_id}")
        logger.info(f"Quota range: {min_quota} to {max_quota}")
        
        # Step 1: Initial predictions with last year's quota
        logger.info("Step 1: Making initial predictions with last year's quota")
        
        initial_base_ranking = self.predict_base_ranking_simple(program_id, prediction_year, current_fee)
        initial_top_ranking = self.predict_top_ranking_simple(program_id, prediction_year, current_fee)
        initial_occupied_predictions = self.predict_occupied_slots_individual(program_id, prediction_year, current_fee)
        
        # Calculate initial weighted average
        initial_weights = self.calculate_dynamic_weights(initial_base_ranking, 0, max_quota - min_quota)
        initial_occupied_avg = sum(w * p for w, p in zip(initial_weights, initial_occupied_predictions))
        
        logger.info(f"Initial base ranking: {initial_base_ranking:.0f}")
        logger.info(f"Initial top ranking: {initial_top_ranking:.0f}")
        logger.info(f"Initial occupied slot predictions: {initial_occupied_predictions}")
        logger.info(f"Initial weights: {[f'{w:.3f}' for w in initial_weights]}")
        logger.info(f"Initial weighted occupied slots: {initial_occupied_avg:.1f}")
        
        # Step 2: Calculate initial Rank_change_per_student
        if initial_occupied_avg > 0:
            initial_rank_change_per_student = (initial_base_ranking - initial_top_ranking) / initial_occupied_avg
        else:
            initial_rank_change_per_student = 0
        
        logger.info(f"Initial Rank_change_per_student: {initial_rank_change_per_student:.2f}")
        
        # Step 3: Dynamic simulation loop
        logger.info("Step 3: Starting dynamic simulation loop")
        
        results = []
        current_base_ranking = initial_base_ranking
        
        total_iterations = max_quota - min_quota + 1
        
        for iteration, quota in enumerate(range(min_quota, max_quota + 1)):
            
            # Calculate dynamic weights for this iteration
            weights = self.calculate_dynamic_weights(current_base_ranking, iteration, total_iterations)
            
            # Get individual model predictions
            occupied_predictions = self.predict_occupied_slots_individual(program_id, prediction_year, quota, current_fee)
            
            # Calculate weighted average and round
            weighted_occupied = sum(w * p for w, p in zip(weights, occupied_predictions))
            rounded_occupied = round(weighted_occupied)
            
            # Check if quota is filled (occupied >= quota)
            quota_filled = rounded_occupied >= quota
            
            # Calculate logarithmic rank change per student for this iteration
            log_factor = math.log(iteration * 2)  # +2 to avoid log(1)=0 and start from log(2)
            current_rank_change_per_student = initial_rank_change_per_student * log_factor
            
            # Calculate ranking stability factor (programs near 0 ranking change slower)
            ranking_stability = compute_ranking_stability(current_base_ranking)
            adjusted_rank_change = current_rank_change_per_student * ranking_stability
            
            occupied_change = rounded_occupied - initial_occupied_avg
            ranking_change = occupied_change * adjusted_rank_change
            current_base_ranking = max(0, initial_base_ranking - ranking_change)
            
            # Calculate revenue
            revenue = rounded_occupied * current_fee
            
            result = {
                'iteration': iteration,
                'quota': quota,
                'weights': weights,
                'individual_predictions': occupied_predictions,
                'weighted_occupied': weighted_occupied,
                'rounded_occupied': rounded_occupied,
                'quota_filled': quota_filled,
                'current_base_ranking': current_base_ranking,
                'rank_change_per_student': current_rank_change_per_student,
                'adjusted_rank_change': adjusted_rank_change,
                'revenue': revenue,
                'ranking_stability': ranking_stability
            }
            
            results.append(result)
            
            logger.info(f"Iteration {iteration+1}: quota={quota}, occupied={rounded_occupied}, "
                       f"ranking={current_base_ranking:.0f}, revenue={revenue:.0f}, filled={quota_filled}")
            
            # Stop conditions
            if not quota_filled:
                prev_occupied = results[iteration-1]['rounded_occupied']
                occupied_change_rate = abs(rounded_occupied - prev_occupied)
                
                # Stop if occupied slots stabilize or ranking becomes too poor
                if occupied_change_rate <= 1 or current_base_ranking > initial_base_ranking * 2:
                    logger.info(f"Stopping optimization: occupied change rate={occupied_change_rate}, "
                               f"ranking deterioration={current_base_ranking/initial_base_ranking:.2f}")
                    break
            if current_base_ranking > threshold:
                logger.info(f"Stopping optimization: base ranking exceeded threshold {threshold}")
                break

        # Find optimal solution
        valid_results = [r for r in results if r['rounded_occupied'] > 0]
        if valid_results:
            optimal = max(valid_results, key=lambda x: x['revenue'])
        else:
            optimal = results[0]
        
        return {
            'program_id': program_id,
            'initial_predictions': {
                'base_ranking': initial_base_ranking,
                'top_ranking': initial_top_ranking,
                'occupied_slots': initial_occupied_avg,
                'rank_change_per_student': initial_rank_change_per_student
            },
            'optimization_results': results,
            'optimal_solution': {
                'quota': optimal['quota'],
                'occupied_slots': optimal['rounded_occupied'],
                'base_ranking': optimal['current_base_ranking'],
                'revenue': optimal['revenue'],
                'weights_used': optimal['weights']
            },
            'improvement': {
                'revenue_increase': optimal['revenue'] - (initial_occupied_avg * current_fee),
                'ranking_change': optimal['current_base_ranking'] - initial_base_ranking
            }
        }

def main():
    """Example usage of the dynamic optimizer"""
    
    # Configuration
    PROJECT_ID = "unioptima-461722"
    DATASET_ID = "university_db"
    SERVICE_ACCOUNT_PATH = "service-account-key.json"
    
    # Initialize optimizer
    optimizer = DynamicRevenueOptimizer(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        service_account_path=SERVICE_ACCOUNT_PATH
    )
    
    # Get user input
    program_id = int(input("Enter program ID: "))
    min_quota = int(input("Enter minimum quota to test: "))
    max_quota = int(input("Enter maximum quota to test: "))
    current_fee = float(input("Enter current fee: "))
    threshold = float(input("Enter ranking threshold (default 100000): ") or 100000)
    
    try:
        results = optimizer.dynamic_optimize_revenue(
            program_id=program_id,
            current_fee=current_fee,
            min_quota=min_quota,
            max_quota=max_quota,
            threshold=threshold
        )

        print("\n" + "="*100)
        print("DYNAMIC REVENUE OPTIMIZATION RESULTS")
        print("="*100)
        
        # Initial predictions
        initial = results['initial_predictions']
        print(f"\nINITIAL PREDICTIONS (with last year's quota={last_year_quota}):")
        print(f"  Base Ranking: {initial['base_ranking']:.0f}")
        print(f"  Top Ranking: {initial['top_ranking']:.0f}")
        print(f"  Occupied Slots: {initial['occupied_slots']:.1f}")
        print(f"  Rank Change per Student: {initial['rank_change_per_student']:.2f}")
        
        # Optimal solution
        optimal = results['optimal_solution']
        print(f"\nOPTIMAL SOLUTION:")
        print(f"  Quota: {optimal['quota']}")
        print(f"  Occupied Slots: {optimal['occupied_slots']}")
        print(f"  Base Ranking: {optimal['base_ranking']:.0f}")
        print(f"  Revenue: {optimal['revenue']:,.0f} TL")
        print(f"  Model Weights: {[f'{w:.3f}' for w in optimal['weights_used']]}")
        
        # Improvements
        improvement = results['improvement']
        print(f"\nIMPROVEMENTS:")
        print(f"  Revenue Increase: {improvement['revenue_increase']:,.0f} TL")
        print(f"  Quota Increase: {improvement['quota_increase']}")
        print(f"  Ranking Change: {improvement['ranking_change']:.0f}")
        
        # Show iteration details
        print(f"\nITERATION DETAILS:")
        for result in results['optimization_results'][-5:]:  # Show last 5 iterations
            print(f"  Quota {result['quota']}: occupied={result['rounded_occupied']}, "
                  f"ranking={result['current_base_ranking']:.0f}, "
                  f"revenue={result['revenue']:,.0f}, "
                  f"filled={result['quota_filled']}")
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}")

if __name__ == "__main__":
    main()