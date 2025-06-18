"""
Advanced Dynamic Revenue Optimization System

This system implements sophisticated three-scenario optimization logic based on the relationship
between user-proposed quota ranges and last year's occupied slots (lag_occupiedSlots).

SCENARIO LOGIC:
1. EXPANSION: min_quota > lag_occupied_slots
   - When predicted > lag_occupied → baseRanking INCREASES (worsens)
   
2. CONTRACTION: max_quota < lag_occupied_slots  
   - When predicted < lag_occupied → baseRanking INCREASES (worsens)
   
3. MIXED: min_quota < lag_occupied_slots < max_quota
   - Phase 1: predicted ≤ lag_occupied → baseRanking DECREASES (improves)
   - Phase 2: predicted > lag_occupied → baseRanking INCREASES (worsens)

MODEL CHARACTERISTICS:
- v1: Trained with all universities (general predictor)
- v2: Trained with unfilled programs (quota-sensitive for poor rankings)  
- v3: No current_quota feature (base demand predictor)
- v4: Minimal features (pure base demand, no quota/lag effects)
"""

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
    """
    Compute ranking stability factor based on program tier.
    
    Zone 1: Elite (0–10,000) → stability ≈ 0.2 (very stable)
    Zone 2: Good (10,000–100,000) → stability ≈ 0.5 (moderate)
    Zone 3: Poor (>100,000) → stability ≈ 1.0 (highly variable)
    """
    # Sigmoid transitions between zones
    transition1 = 1 / (1 + math.exp(-(current_rank - 10000) / 3000))
    transition2 = 1 / (1 + math.exp(-(current_rank - 100000) / 20000))
    
    # Combined stability factor
    stability = 0.2 + (0.3 * transition1) + (0.5 * transition2)
    return min(stability, 1.0)


class AdvancedDynamicRevenueOptimizer:
    def __init__(self, 
                 project_id: str,
                 dataset_id: str,
                 service_account_path: str,
                 models_dir: str = "models",
                 scalers_dir: str = "scalers"):
        """Initialize the advanced dynamic revenue optimizer."""
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
        
        # Load all models and scalers
        self._load_models()
        
    def _load_models(self):
        """Load all required models and scalers."""
        try:
            # Load occupied slot models and scalers
            self.occupied_slot_models = []
            self.occupied_slot_scalers = []
            self.occupied_slot_model_configs = []
            
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

    def calculate_tier_based_weights(self, base_ranking: float, iteration: int, total_iterations: int) -> List[float]:
        """
        Calculate dynamic weights based on program tier and iteration progress.
        
        Program Tiers and Model Usage:
        - Elite (0-1K): v1 dominates (always fill quota)
        - Good (1K-10K): v1 + some v2 (quota sensitive)  
        - Average (10K-100K): Balanced (moderate quota effects)
        - Poor (100K+): v2 dominates early, v3/v4 later (saturation)
        """
        progress = iteration / total_iterations if total_iterations > 0 else 0
        
        # Determine program tier
        if base_ranking <= 1000:
            tier = "elite"
        elif base_ranking <= 10000:
            tier = "good"
        elif base_ranking <= 100000:
            tier = "average"
        else:
            tier = "poor"
        
        if tier == "elite":
            # Elite programs: v1 dominates, minimal quota sensitivity
            v1_weight = 0.70 - (progress * 0.10)  # High but slightly decreases
            v2_weight = 0.05 + (progress * 0.05)  # Minimal quota effect
            v3_weight = 0.15 + (progress * 0.10)  # Base demand grows
            v4_weight = 0.10 + (progress * 0.10)  # Pure demand grows
            
        elif tier == "good":
            # Good programs: High v1, some quota sensitivity
            quota_sensitivity = 1 - progress
            v1_weight = 0.50 + (quota_sensitivity * 0.20)
            v2_weight = 0.15 + (quota_sensitivity * 0.15)
            v3_weight = 0.20 + (progress * 0.15)
            v4_weight = 0.15 + (progress * 0.10)
            
        elif tier == "average":
            # Average programs: Balanced approach
            quota_sensitivity = (1 - progress) * 0.8
            ranking_factor = min((base_ranking - 10000) / 90000, 1.0)
            
            v1_weight = 0.40 + (quota_sensitivity * 0.10)
            v2_weight = 0.20 + (quota_sensitivity * 0.20) + (ranking_factor * 0.10)
            v3_weight = 0.25 + (progress * 0.15)
            v4_weight = 0.15 + (progress * 0.10)
            
        else:  # tier == "poor"
            # Poor programs: High v2 early, then saturation to v3/v4
            ranking_factor = min((base_ranking - 100000) / 400000, 1.0)
            saturation_effect = math.pow(progress, 2)  # Accelerating saturation
            
            v1_weight = 0.30 - (saturation_effect * 0.10)
            v2_weight = 0.40 + (ranking_factor * 0.20) - (saturation_effect * 0.40)
            v3_weight = 0.15 + (saturation_effect * 0.30)
            v4_weight = 0.15 + (saturation_effect * 0.25)
        
        # Compile and normalize weights
        weights = [v1_weight, v2_weight, v3_weight, v4_weight]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Apply minimum thresholds
        min_weights = [0.08, 0.05, 0.05, 0.05]
        weights = [max(w, min_w) for w, min_w in zip(weights, min_weights)]
        
        # Final normalization
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return weights

    def predict_occupied_slots_individual(self, program_id: int, prediction_year: int, quota: int = None, fee: float = None) -> List[float]:
        """
        Predict occupied slots using individual models.
        
        Returns:
            List of predictions from [v1, v2, v3, v4] models
        """
        predictions = []
        
        for i, (model, scaler, config) in enumerate(zip(self.occupied_slot_models, self.occupied_slot_scalers, self.occupied_slot_model_configs)):
            # Get base input vectors for this specific model version
            base_input = self.get_base_input_vectors([program_id], prediction_year, f'occupied_slot_v{i+1}')

            if base_input.empty:
                logger.error(f"No base input found for program {program_id} with model v{i+1}")
                predictions.append(0)
                continue
            
            # Make a copy to avoid modifying original data
            input_df = base_input.copy()
            
            # Override quota if provided (only for models that have quota features)
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
        
        if base_input.empty:
            raise ValueError(f"No base input vectors found for base ranking model and program {program_id}")
        
        input_df = base_input.copy()

        # Only override fee if provided and column exists
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

    def predict_top_ranking_simple(self, program_id: int, prediction_year: int, fee: float = None) -> float:
        """Simplified top ranking prediction."""
        base_input = self.get_base_input_vectors([program_id], prediction_year, 'top_ranking')

        if base_input.empty:
            raise ValueError(f"No base input vectors found for top ranking model and program {program_id}")
        
        input_df = base_input.copy()

        # Only override fee if provided and column exists
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

    def calculate_ranking_change(self, scenario: str, rounded_occupied: int, lag_occupied_slots: int, 
                               base_rank_change_per_student: float, current_base_ranking: float,
                               initial_base_ranking: float, counter: int) -> float:
        """
        Calculate ranking change based on scenario logic - CORE RANKING LOGIC
        This is the consistent ranking calculation used across all optimization methods.
        """
        
        if scenario == "EXPANSION":
            # EXPANSION LOGIC: When predicted > lag_occupied → ranking worsens
            if rounded_occupied > lag_occupied_slots:
                # Calculate diminishing logarithmic effect
                log_factor = math.log(counter + 1)
                rank_change_magnitude = abs(base_rank_change_per_student) * log_factor
                
                # Apply program-specific stability factor
                stability = compute_ranking_stability(current_base_ranking)
                adjusted_change = rank_change_magnitude * stability
                
                # Calculate ranking deterioration (increase = worse)
                excess_students = rounded_occupied - lag_occupied_slots
                ranking_deterioration = excess_students * adjusted_change
                return initial_base_ranking + ranking_deterioration
            else:
                # Predicted ≤ lag_occupied → minimal ranking change
                return initial_base_ranking
                
        elif scenario == "CONTRACTION":
            # CONTRACTION LOGIC: When predicted < lag_occupied → ranking worsens
            if rounded_occupied < lag_occupied_slots:
                # Calculate diminishing logarithmic effect
                log_factor = math.log(counter + 1)
                rank_change_magnitude = abs(base_rank_change_per_student) / log_factor
                
                # Apply program-specific stability factor
                stability = compute_ranking_stability(current_base_ranking)
                adjusted_change = rank_change_magnitude * stability
                
                # Calculate ranking deterioration due to demand drop
                deficit_students = lag_occupied_slots - rounded_occupied
                ranking_deterioration = deficit_students * adjusted_change * 0.3
                return initial_base_ranking + ranking_deterioration
            else:
                # Predicted ≥ lag_occupied → minimal ranking change
                return initial_base_ranking
                
        elif scenario == "MIXED_IMPROVE":
            # MIXED LOGIC - Phase 1: More selective than last year → ranking improves
            # Calculate diminishing logarithmic effect
            log_factor = math.log(counter + 1)
            rank_change_magnitude = abs(base_rank_change_per_student) / log_factor
            
            # Apply program-specific stability factor
            stability = compute_ranking_stability(current_base_ranking)
            adjusted_change = rank_change_magnitude * stability
            
            # Calculate ranking improvement (decrease = better)
            selectivity_increase = lag_occupied_slots - rounded_occupied
            ranking_improvement = selectivity_increase * adjusted_change * 0.4
            return max(0, initial_base_ranking - ranking_improvement)
            
        elif scenario == "MIXED_WORSEN":
            # MIXED LOGIC - Phase 2: Less selective than last year → ranking worsens
            # Calculate diminishing logarithmic effect
            log_factor = math.log(counter + 1)
            rank_change_magnitude = abs(base_rank_change_per_student) / log_factor
            
            # Apply program-specific stability factor
            stability = compute_ranking_stability(current_base_ranking)
            adjusted_change = rank_change_magnitude * stability
            
            # Calculate ranking deterioration
            excess_students = rounded_occupied - lag_occupied_slots
            ranking_deterioration = excess_students * adjusted_change * 0.5
            return initial_base_ranking + ranking_deterioration
        
        else:
            # Default case
            return current_base_ranking

    def dynamic_optimize_revenue(self, 
                                program_id: int,
                                current_fee: float,
                                min_quota: int,
                                max_quota: int,
                                ranking_threshold: float = 500000,
                                prediction_year: int = 2025) -> Dict:
        """
        Advanced dynamic optimization with three distinct scenarios and separate iteration counters.
        """
        
        logger.info(f"🚀 Starting Advanced Dynamic Optimization for Program {program_id}")
        logger.info(f"📊 Quota Range: {min_quota} to {max_quota}")
        logger.info(f"💰 Fee: {current_fee:,.0f} TL")
        
        # ================================================================================
        # STEP 1: BASELINE PREDICTIONS AND DATA EXTRACTION
        # ================================================================================
        logger.info("\n📋 STEP 1: Extracting baseline data and predictions")
        
        # PREDICT ONCE: Get baseline predictions (FIXED LOGIC)
        initial_base_ranking = self.predict_base_ranking_simple(program_id, prediction_year, current_fee)
        initial_top_ranking = self.predict_top_ranking_simple(program_id, prediction_year, current_fee)
        initial_occupied_predictions = self.predict_occupied_slots_individual(program_id, prediction_year, None, current_fee)
        
        # Extract last year's occupied slots from historical data
        base_input = self.get_base_input_vectors([program_id], prediction_year, 'occupied_slot_v1')
        if base_input.empty or 'lag_occupiedSlots' not in base_input.columns:
            raise ValueError(f"Cannot find lag_occupiedSlots for program {program_id}")
        
        lag_occupied_slots = int(base_input['lag_occupiedSlots'].iloc[0])
        
        # Calculate initial weighted prediction
        initial_weights = self.calculate_tier_based_weights(initial_base_ranking, 0, max_quota - min_quota)
        initial_occupied_avg = sum(w * p for w, p in zip(initial_weights, initial_occupied_predictions))
        
        # CALCULATE ONCE: base_rank_change_per_student (FIXED LOGIC)
        if initial_occupied_avg > 0:
            base_rank_change_per_student = (initial_base_ranking - initial_top_ranking) / initial_occupied_avg
        else:
            base_rank_change_per_student = 0
        
        logger.info(f"📈 Last Year's Occupied Slots: {lag_occupied_slots}")
        logger.info(f"🏆 Initial Base Ranking: {initial_base_ranking:,.0f}")
        logger.info(f"🥇 Initial Top Ranking: {initial_top_ranking:,.0f}")
        logger.info(f"👥 Initial Predicted Occupied: {initial_occupied_avg:.1f}")
        logger.info(f"📊 Base Rank Change per Student: {base_rank_change_per_student:.2f}")
        
        # ================================================================================
        # STEP 2: SCENARIO CLASSIFICATION
        # ================================================================================
        logger.info("\n🔍 STEP 2: Determining optimization scenario")
        
        if min_quota > lag_occupied_slots:
            scenario = "EXPANSION"
            logger.info(f"📈 EXPANSION SCENARIO: min_quota ({min_quota}) > lag_occupied ({lag_occupied_slots})")
        elif max_quota < lag_occupied_slots:
            scenario = "CONTRACTION"
            logger.info(f"📉 CONTRACTION SCENARIO: max_quota ({max_quota}) < lag_occupied ({lag_occupied_slots})")
        else:
            scenario = "MIXED"
            logger.info(f"🔄 MIXED SCENARIO: min_quota ({min_quota}) < lag_occupied ({lag_occupied_slots}) < max_quota ({max_quota})")
        
        # ================================================================================
        # STEP 3: INITIALIZE TRACKING VARIABLES
        # ================================================================================
        logger.info("\n⚙️ STEP 3: Initializing scenario-specific counters and variables")
        
        # Initialize separate iteration counters for each scenario condition
        expansion_iterations = 0          # Counts when predicted > lag in expansion
        contraction_iterations = 0        # Counts when predicted < lag in contraction  
        mixed_improve_iterations = 0      # Counts when predicted ≤ lag in mixed
        mixed_worsen_iterations = 0       # Counts when predicted > lag in mixed
        mixed_crossover_point = None      # Tracks where crossover occurs
        
        # Initialize results tracking
        results = []
        current_base_ranking = initial_base_ranking
        total_iterations = max_quota - min_quota + 1
        
        logger.info(f"🔄 Total Iterations Planned: {total_iterations}")
        logger.info(f"🎯 Ranking Threshold: {ranking_threshold:,.0f}")
        
        # ================================================================================
        # STEP 4: MAIN OPTIMIZATION LOOP WITH SCENARIO-SPECIFIC LOGIC
        # ================================================================================
        logger.info("\n🔄 STEP 4: Starting main optimization loop")
        
        for iteration, quota in enumerate(range(min_quota, max_quota + 1)):
            logger.info(f"\n--- Iteration {iteration + 1}/{total_iterations}: Testing Quota {quota} ---")
            
            # Calculate dynamic model weights for current iteration
            weights = self.calculate_tier_based_weights(current_base_ranking, iteration, total_iterations)
            
            # Get individual model predictions for current quota
            occupied_predictions = self.predict_occupied_slots_individual(program_id, prediction_year, quota, current_fee)
            
            # Calculate weighted average and round to whole students
            weighted_occupied = sum(w * p for w, p in zip(weights, occupied_predictions))
            rounded_occupied = round(weighted_occupied)
            quota_filled = rounded_occupied >= quota
            
            logger.info(f"👥 Final Prediction: {rounded_occupied} students")
            logger.info(f"✅ Quota Filled: {'Yes' if quota_filled else 'No'}")
            
            # ========================================================================
            # STEP 4A: APPLY SCENARIO-SPECIFIC RANKING LOGIC (FIXED LOGIC)
            # ========================================================================
            
            if scenario == "EXPANSION":
                # EXPANSION LOGIC: When predicted > lag_occupied → ranking worsens
                if rounded_occupied > lag_occupied_slots:
                    expansion_iterations += 1
                    current_base_ranking = self.calculate_ranking_change(
                        "EXPANSION", rounded_occupied, lag_occupied_slots, 
                        base_rank_change_per_student, current_base_ranking, 
                        initial_base_ranking, expansion_iterations
                    )
                    logger.info(f"📈 EXPANSION: predicted ({rounded_occupied}) > lag ({lag_occupied_slots})")
                    logger.info(f"🔢 Expansion iterations: {expansion_iterations}")
                else:
                    current_base_ranking = initial_base_ranking
                    logger.info(f"➡️ EXPANSION: predicted ({rounded_occupied}) ≤ lag ({lag_occupied_slots}) → No ranking change")
                    
            elif scenario == "CONTRACTION":
                # CONTRACTION LOGIC: When predicted < lag_occupied → ranking worsens
                if rounded_occupied < lag_occupied_slots:
                    contraction_iterations += 1
                    current_base_ranking = self.calculate_ranking_change(
                        "CONTRACTION", rounded_occupied, lag_occupied_slots, 
                        base_rank_change_per_student, current_base_ranking, 
                        initial_base_ranking, contraction_iterations
                    )
                    logger.info(f"📉 CONTRACTION: predicted ({rounded_occupied}) < lag ({lag_occupied_slots})")
                    logger.info(f"🔢 Contraction iterations: {contraction_iterations}")
                else:
                    current_base_ranking = initial_base_ranking
                    logger.info(f"➡️ CONTRACTION: predicted ({rounded_occupied}) ≥ lag ({lag_occupied_slots}) → No ranking change")
                    
            else:  # scenario == "MIXED"
                # MIXED LOGIC: Two distinct phases with crossover detection
                if rounded_occupied <= lag_occupied_slots:
                    # PHASE 1: More selective than last year → ranking improves
                    mixed_improve_iterations += 1
                    current_base_ranking = self.calculate_ranking_change(
                        "MIXED_IMPROVE", rounded_occupied, lag_occupied_slots, 
                        base_rank_change_per_student, current_base_ranking, 
                        initial_base_ranking, mixed_improve_iterations
                    )
                    logger.info(f"⬇️ MIXED-IMPROVE: predicted ({rounded_occupied}) ≤ lag ({lag_occupied_slots})")
                    logger.info(f"🔢 Mixed improve iterations: {mixed_improve_iterations}")
                    
                else:
                    # PHASE 2: Less selective than last year → ranking worsens
                    if mixed_crossover_point is None:
                        mixed_crossover_point = quota
                        logger.info(f"🎯 MIXED CROSSOVER DETECTED at quota {quota}!")
                    
                    mixed_worsen_iterations += 1
                    current_base_ranking = self.calculate_ranking_change(
                        "MIXED_WORSEN", rounded_occupied, lag_occupied_slots, 
                        base_rank_change_per_student, current_base_ranking, 
                        initial_base_ranking, mixed_worsen_iterations
                    )
                    logger.info(f"⬆️ MIXED-WORSEN: predicted ({rounded_occupied}) > lag ({lag_occupied_slots})")
                    logger.info(f"🔢 Mixed worsen iterations: {mixed_worsen_iterations}")
            
            logger.info(f"🏆 Current Base Ranking: {current_base_ranking:,.0f}")
            
            # Calculate revenue for this iteration
            actual_occupied = min(quota, rounded_occupied)
            revenue = actual_occupied * current_fee
            logger.info(f"💰 Revenue: {revenue:,.0f} TL")
            
            # ========================================================================
            # STEP 4B: STORE ITERATION RESULTS
            # ========================================================================
            result = {
                'iteration': iteration,
                'quota': quota,
                'weights': weights,
                'individual_predictions': occupied_predictions,
                'weighted_occupied': weighted_occupied,
                'rounded_occupied': rounded_occupied,
                'actual_occupied': actual_occupied,
                'quota_filled': quota_filled,
                'current_base_ranking': current_base_ranking,
                'revenue': revenue,
                'lag_occupied_slots': lag_occupied_slots,
                'scenario': scenario,
                'expansion_iterations': expansion_iterations,
                'contraction_iterations': contraction_iterations,
                'mixed_improve_iterations': mixed_improve_iterations,
                'mixed_worsen_iterations': mixed_worsen_iterations,
                'crossover_point': mixed_crossover_point
            }
            results.append(result)
            
            # ========================================================================
            # STEP 4C: APPLY DYNAMIC STOPPING CONDITIONS
            # ========================================================================
            
            # PRIORITY 1: Check ranking deterioration (highest priority)
            ranking_deterioration_ratio = current_base_ranking / initial_base_ranking
            if current_base_ranking > ranking_threshold:
                logger.info(f"🛑 STOPPING (Priority 1): Ranking exceeded threshold")
                break
            elif ranking_deterioration_ratio > 3.0:
                logger.info(f"🛑 STOPPING (Priority 1): Ranking deteriorated too much")
                break
            
            # PRIORITY 2: If quota is filled, ALWAYS continue 
            if quota_filled:
                logger.info(f"✅ CONTINUING: Quota filled - ignoring other stop conditions")
                continue
            
            # PRIORITY 3: Dynamic occupiedSlot stability analysis
            if iteration >= 2:
                recent_occupied = [results[i]['rounded_occupied'] for i in range(max(0, iteration-4), iteration+1)]
                recent_quotas = [results[i]['quota'] for i in range(max(0, iteration-4), iteration+1)]
                
                occupied_changes = [abs(recent_occupied[i] - recent_occupied[i-1]) for i in range(1, len(recent_occupied))]
                max_stable_iterations = min(8, total_iterations // 3)
                consecutive_stable = 0
                
                for i in range(len(occupied_changes)-1, -1, -1):
                    if occupied_changes[i] <= 1:
                        consecutive_stable += 1
                    else:
                        break
                
                if len(recent_occupied) >= 5:
                    unique_occupied = len(set(recent_occupied))
                    quota_range = max(recent_quotas) - min(recent_quotas)
                    
                    if unique_occupied <= 2 and quota_range >= 5:
                        remaining_iterations = total_iterations - iteration - 1
                        if consecutive_stable >= max_stable_iterations and remaining_iterations < 3:
                            logger.info(f"🛑 STOPPING (Priority 3): Plateau confirmed")
                            break
                
                elif consecutive_stable >= max_stable_iterations:
                    recent_revenues = [results[i]['revenue'] for i in range(max(0, iteration-2), iteration+1)]
                    revenue_improving = len(recent_revenues) >= 3 and recent_revenues[-1] > recent_revenues[0]
                    
                    if not revenue_improving:
                        logger.info(f"🛑 STOPPING (Priority 3): OccupiedSlot stable, no revenue improvement")
                        break
        
        # ================================================================================
        # STEP 5: FIND OPTIMAL SOLUTION AND COMPILE RESULTS
        # ================================================================================
        logger.info("\n🏆 STEP 5: Finding optimal solution")
        
        valid_results = [r for r in results if r['rounded_occupied'] > 0]
        if valid_results:
            optimal = max(valid_results, key=lambda x: x['revenue'])
        else:
            optimal = results[0]
        
        logger.info(f"🎯 Optimal Solution Found:")
        logger.info(f"  📊 Quota: {optimal['quota']}")
        logger.info(f"  👥 Occupied Slots: {optimal['actual_occupied']}")
        logger.info(f"  🏆 Base Ranking: {optimal['current_base_ranking']:,.0f}")
        logger.info(f"  💰 Revenue: {optimal['revenue']:,.0f} TL")
        
        return {
            'program_id': program_id,
            'scenario': scenario,
            'lag_occupied_slots': lag_occupied_slots,
            'initial_predictions': {
                'base_ranking': initial_base_ranking,
                'top_ranking': initial_top_ranking,
                'occupied_slots': initial_occupied_avg,
                'rank_change_per_student': base_rank_change_per_student
            },
            'optimization_results': results,
            'optimal_solution': {
                'quota': optimal['quota'],
                'occupied_slots': optimal['actual_occupied'],
                'base_ranking': optimal['current_base_ranking'],
                'revenue': optimal['revenue'],
                'weights_used': optimal['weights']
            },
            'improvements': {
                'revenue_increase': optimal['revenue'] - (initial_occupied_avg * current_fee),
                'ranking_change': optimal['current_base_ranking'] - initial_base_ranking,
                'quota_change': optimal['quota'] - lag_occupied_slots
            },
            'scenario_statistics': {
                'expansion_iterations': expansion_iterations,
                'contraction_iterations': contraction_iterations,
                'mixed_improve_iterations': mixed_improve_iterations,
                'mixed_worsen_iterations': mixed_worsen_iterations,
                'crossover_point': mixed_crossover_point,
                'total_iterations_run': len(results)
            }
        }

    def optimize_fee_with_fixed_quota(self, 
                                     program_id: int,
                                     fixed_quota: int,
                                     min_fee: float,
                                     max_fee: float,
                                     fee_step: float = 5000,
                                     ranking_threshold: float = 500000,
                                     max_ranking_deterioration_pct: float = 20,
                                     prediction_year: int = 2025) -> Dict:
        """
        TASK 3: Fee Optimization with Fixed Quota
        Uses SAME baseRanking logic as Task 2, but iterates over fee range instead of quota range.
        """
        
        logger.info(f"🎓 Starting Task 3: Fee Optimization with Fixed Quota for Program {program_id}")
        logger.info(f"📊 Fixed Quota: {fixed_quota} students")
        logger.info(f"💰 Fee Range: {min_fee:,.0f} to {max_fee:,.0f} TL (step: {fee_step:,.0f})")
        logger.info(f"🎯 Max Ranking Deterioration: {max_ranking_deterioration_pct}%")
        
        # ================================================================================
        # STEP 1: BASELINE PREDICTIONS AND DATA EXTRACTION (SAME AS TASK 2)
        # ================================================================================
        logger.info("\n📋 STEP 1: Extracting baseline data and predictions")
        
        # Use middle fee for baseline predictions
        baseline_fee = (min_fee + max_fee) / 2
        
        # Get baseline predictions (SAME LOGIC AS TASK 2)
        initial_base_ranking = self.predict_base_ranking_simple(program_id, prediction_year, baseline_fee)
        initial_top_ranking = self.predict_top_ranking_simple(program_id, prediction_year, baseline_fee)
        initial_occupied_predictions = self.predict_occupied_slots_individual(program_id, prediction_year, fixed_quota, baseline_fee)
        
        # Extract last year's occupied slots from historical data (SAME AS TASK 2)
        base_input = self.get_base_input_vectors([program_id], prediction_year, 'occupied_slot_v1')
        if base_input.empty or 'lag_occupiedSlots' not in base_input.columns:
            raise ValueError(f"Cannot find lag_occupiedSlots for program {program_id}")
        
        lag_occupied_slots = int(base_input['lag_occupiedSlots'].iloc[0])
        
        # Calculate initial weighted prediction (SAME AS TASK 2)
        initial_weights = self.calculate_tier_based_weights(initial_base_ranking, 0, 1)
        initial_occupied_avg = sum(w * p for w, p in zip(initial_weights, initial_occupied_predictions))
        
        # Calculate baseline rank change per student (SAME AS TASK 2)
        if initial_occupied_avg > 0:
            base_rank_change_per_student = (initial_base_ranking - initial_top_ranking) / initial_occupied_avg
        else:
            base_rank_change_per_student = 0
        
        # Set ranking constraints
        max_allowed_ranking = min(
            ranking_threshold,
            initial_base_ranking * (1 + max_ranking_deterioration_pct / 100)
        )
        
        logger.info(f"📈 Last Year's Occupied Slots: {lag_occupied_slots}")
        logger.info(f"🏆 Initial Base Ranking: {initial_base_ranking:,.0f}")
        logger.info(f"📊 Base Rank Change per Student: {base_rank_change_per_student:.2f}")
        logger.info(f"🎯 Maximum Allowed Ranking: {max_allowed_ranking:,.0f}")
        
        # ================================================================================
        # STEP 2: DETERMINE QUOTA SCENARIO (SAME LOGIC AS TASK 2)
        # ================================================================================
        logger.info("\n🔍 STEP 2: Determining quota scenario for fixed quota")
        
        if fixed_quota > lag_occupied_slots:
            scenario = "EXPANSION"
            logger.info(f"📈 EXPANSION: Fixed quota ({fixed_quota}) > lag_occupied ({lag_occupied_slots})")
        elif fixed_quota < lag_occupied_slots:
            scenario = "CONTRACTION"
            logger.info(f"📉 CONTRACTION: Fixed quota ({fixed_quota}) < lag_occupied ({lag_occupied_slots})")
        else:
            scenario = "STABLE"
            logger.info(f"➡️ STABLE: Fixed quota ({fixed_quota}) = lag_occupied ({lag_occupied_slots})")
        
        # ================================================================================
        # STEP 3: FEE OPTIMIZATION LOOP (ADAPTED FROM TASK 2)
        # ================================================================================
        logger.info("\n🔄 STEP 3: Starting fee optimization iterations")
        
        # Initialize tracking variables (SAME AS TASK 2)
        expansion_iterations = 0
        contraction_iterations = 0
        mixed_improve_iterations = 0
        mixed_worsen_iterations = 0
        mixed_crossover_point = None
        
        results = []
        current_base_ranking = initial_base_ranking
        fee_range = np.arange(min_fee, max_fee + fee_step, fee_step)
        total_iterations = len(fee_range)
        
        logger.info(f"🔄 Total Fee Iterations Planned: {total_iterations}")
        logger.info(f"🎯 Ranking Threshold: {ranking_threshold:,.0f}")
        
        for iteration, current_fee in enumerate(fee_range):
            logger.info(f"\n--- Fee Iteration {iteration + 1}/{total_iterations}: Testing Fee {current_fee:,.0f} TL ---")
            
            # Calculate dynamic model weights for current iteration (SAME AS TASK 2)
            weights = self.calculate_tier_based_weights(current_base_ranking, iteration, total_iterations)
            logger.info(f"⚖️ Model Weights: v1={weights[0]:.3f}, v2={weights[1]:.3f}, v3={weights[2]:.3f}, v4={weights[3]:.3f}")
            
            # Get individual model predictions for fixed quota and current fee
            occupied_predictions = self.predict_occupied_slots_individual(program_id, prediction_year, fixed_quota, current_fee)
            logger.info(f"🔮 Raw Predictions: v1={occupied_predictions[0]:.1f}, v2={occupied_predictions[1]:.1f}, v3={occupied_predictions[2]:.1f}, v4={occupied_predictions[3]:.1f}")
            
            # Calculate weighted average and round to whole students
            weighted_occupied = sum(w * p for w, p in zip(weights, occupied_predictions))
            rounded_occupied = round(weighted_occupied)
            quota_filled = rounded_occupied >= fixed_quota
            
            logger.info(f"👥 Final Prediction: {rounded_occupied} students (weighted: {weighted_occupied:.1f})")
            logger.info(f"✅ Quota Filled: {'Yes' if quota_filled else 'No'}")
            
            # ========================================================================
            # APPLY SAME RANKING LOGIC AS TASK 2
            # ========================================================================
            
            if scenario == "EXPANSION":
                # EXPANSION LOGIC: When predicted > lag_occupied → ranking worsens (SAME AS TASK 2)
                if rounded_occupied > lag_occupied_slots:
                    expansion_iterations += 1
                    log_factor = math.log(expansion_iterations + 1)
                    rank_change_magnitude = abs(base_rank_change_per_student) * log_factor
                    stability = compute_ranking_stability(current_base_ranking)
                    adjusted_change = rank_change_magnitude * stability
                    excess_students = rounded_occupied - lag_occupied_slots
                    ranking_deterioration = excess_students * adjusted_change
                    current_base_ranking = initial_base_ranking + ranking_deterioration
                    
                    logger.info(f"📈 EXPANSION: predicted ({rounded_occupied}) > lag ({lag_occupied_slots})")
                    logger.info(f"📊 Ranking worsens by {ranking_deterioration:.0f} → New ranking: {current_base_ranking:.0f}")
                    logger.info(f"🔢 Expansion iterations: {expansion_iterations}")
                else:
                    current_base_ranking = initial_base_ranking
                    logger.info(f"➡️ EXPANSION: predicted ({rounded_occupied}) ≤ lag ({lag_occupied_slots}) → No ranking change")
                    
            elif scenario == "CONTRACTION":
                # CONTRACTION LOGIC: When predicted < lag_occupied → ranking worsens (SAME AS TASK 2)
                if rounded_occupied < lag_occupied_slots:
                    contraction_iterations += 1
                    log_factor = math.log(contraction_iterations + 1)
                    rank_change_magnitude = abs(base_rank_change_per_student) / log_factor
                    stability = compute_ranking_stability(current_base_ranking)
                    adjusted_change = rank_change_magnitude * stability
                    deficit_students = lag_occupied_slots - rounded_occupied
                    ranking_deterioration = deficit_students * adjusted_change * 0.3
                    current_base_ranking = initial_base_ranking + ranking_deterioration
                    
                    logger.info(f"📉 CONTRACTION: predicted ({rounded_occupied}) < lag ({lag_occupied_slots})")
                    logger.info(f"📊 Ranking worsens by {ranking_deterioration:.0f} → New ranking: {current_base_ranking:.0f}")
                    logger.info(f"🔢 Contraction iterations: {contraction_iterations}")
                else:
                    current_base_ranking = initial_base_ranking
                    logger.info(f"➡️ CONTRACTION: predicted ({rounded_occupied}) ≥ lag ({lag_occupied_slots}) → No ranking change")
            else:  # STABLE
                current_base_ranking = initial_base_ranking
                logger.info(f"➡️ STABLE: No ranking change for stable quota")
            
            # Calculate revenue for this iteration
            actual_occupied = min(fixed_quota, rounded_occupied)
            revenue = actual_occupied * current_fee
            
            # Check ranking constraint
            ranking_violation = current_base_ranking > max_allowed_ranking
            ranking_deterioration_pct = ((current_base_ranking - initial_base_ranking) / initial_base_ranking) * 100
            
            logger.info(f"💰 Revenue: {revenue:,.0f} TL")
            logger.info(f"⚠️ Ranking Violation: {'Yes' if ranking_violation else 'No'}")
            
            # Store results
            result = {
                'iteration': iteration,
                'fee': current_fee,
                'fixed_quota': fixed_quota,
                'weights': weights,
                'individual_predictions': occupied_predictions,
                'weighted_occupied': weighted_occupied,
                'rounded_occupied': rounded_occupied,
                'actual_occupied': actual_occupied,
                'quota_filled': quota_filled,
                'current_base_ranking': current_base_ranking,
                'ranking_deterioration_pct': ranking_deterioration_pct,
                'ranking_violation': ranking_violation,
                'revenue': revenue,
                'scenario': scenario,
                'lag_occupied_slots': lag_occupied_slots,
                'expansion_iterations': expansion_iterations,
                'contraction_iterations': contraction_iterations
            }
            results.append(result)
            
            # ========================================================================
            # APPLY SAME DYNAMIC STOPPING CONDITIONS AS TASK 2
            # ========================================================================
            
            # PRIORITY 1: Check ranking deterioration (SAME AS TASK 2)
            ranking_deterioration_ratio = current_base_ranking / initial_base_ranking
            if current_base_ranking > ranking_threshold:
                logger.info(f"🛑 STOPPING (Priority 1): Ranking exceeded threshold")
                break
            elif ranking_deterioration_ratio > 3.0:
                logger.info(f"🛑 STOPPING (Priority 1): Ranking deteriorated too much")
                break
            
            # PRIORITY 2: If quota is filled, ALWAYS continue (SAME AS TASK 2)
            if quota_filled:
                logger.info(f"✅ CONTINUING: Quota filled - testing higher fees")
                continue
            
            # PRIORITY 3: Dynamic occupiedSlot stability analysis (SAME AS TASK 2)
            if iteration >= 2:
                recent_occupied = [results[i]['rounded_occupied'] for i in range(max(0, iteration-4), iteration+1)]
                recent_fees = [results[i]['fee'] for i in range(max(0, iteration-4), iteration+1)]
                
                occupied_changes = [abs(recent_occupied[i] - recent_occupied[i-1]) for i in range(1, len(recent_occupied))]
                max_stable_iterations = min(8, total_iterations // 3)
                consecutive_stable = 0
                
                for i in range(len(occupied_changes)-1, -1, -1):
                    if occupied_changes[i] <= 1:
                        consecutive_stable += 1
                    else:
                        break
                
                if len(recent_occupied) >= 5:
                    unique_occupied = len(set(recent_occupied))
                    fee_range_tested = max(recent_fees) - min(recent_fees)
                    
                    if unique_occupied <= 2 and fee_range_tested >= fee_step * 3:
                        remaining_iterations = total_iterations - iteration - 1
                        if consecutive_stable >= max_stable_iterations and remaining_iterations < 3:
                            logger.info(f"🛑 STOPPING (Priority 3): Fee plateau confirmed")
                            break
                
                elif consecutive_stable >= max_stable_iterations:
                    recent_revenues = [results[i]['revenue'] for i in range(max(0, iteration-2), iteration+1)]
                    revenue_improving = len(recent_revenues) >= 3 and recent_revenues[-1] > recent_revenues[0]
                    
                    if not revenue_improving:
                        logger.info(f"🛑 STOPPING (Priority 3): OccupiedSlot stable, no revenue improvement")
                        break
        
        # ================================================================================
        # STEP 4: FIND OPTIMAL SOLUTION (SAME AS TASK 2)
        # ================================================================================
        logger.info("\n🏆 STEP 4: Finding optimal fee solution")
        
        valid_results = [r for r in results if not r['ranking_violation']]
        
        if valid_results:
            optimal = max(valid_results, key=lambda x: x['revenue'])
            logger.info(f"🎯 Optimal Solution Found: Fee {optimal['fee']:,.0f} TL, Revenue {optimal['revenue']:,.0f} TL")
        else:
            optimal = min(results, key=lambda x: x['ranking_deterioration_pct'])
            logger.info(f"⚠️ No fully compliant solution found")
        
        return {
            'program_id': program_id,
            'optimization_type': 'task3_fee_optimization',
            'fixed_quota': fixed_quota,
            'scenario': scenario,
            'lag_occupied_slots': lag_occupied_slots,
            'fee_range': {'min': min_fee, 'max': max_fee, 'step': fee_step},
            'constraints': {
                'ranking_threshold': ranking_threshold,
                'max_ranking_deterioration_pct': max_ranking_deterioration_pct,
                'max_allowed_ranking': max_allowed_ranking
            },
            'initial_predictions': {
                'base_ranking': initial_base_ranking,
                'top_ranking': initial_top_ranking,
                'occupied_slots': initial_occupied_avg,
                'rank_change_per_student': base_rank_change_per_student
            },
            'optimization_results': results,
            'optimal_solution': {
                'fee': optimal['fee'],
                'quota': fixed_quota,
                'occupied_slots': optimal['actual_occupied'],
                'base_ranking': optimal['current_base_ranking'],
                'revenue': optimal['revenue'],
                'quota_filled': optimal['quota_filled'],
                'ranking_violation': optimal['ranking_violation'],
                'weights_used': optimal['weights']
            },
            'improvements': {
                'revenue_increase': optimal['revenue'] - (initial_occupied_avg * min_fee),
                'ranking_change': optimal['current_base_ranking'] - initial_base_ranking
            },
            'scenario_statistics': {
                'expansion_iterations': expansion_iterations,
                'contraction_iterations': contraction_iterations,
                'total_iterations_run': len(results)
            }
        }

    def optimize_quota_and_fee_combination(self, 
                                         program_id: int,
                                         min_quota: int,
                                         max_quota: int,
                                         min_fee: float,
                                         max_fee: float,
                                         quota_step: int = 5,
                                         fee_step: float = 10000,
                                         ranking_threshold: float = 500000,
                                         max_ranking_deterioration_pct: float = 25,
                                         prediction_year: int = 2025) -> Dict:
        """
        TASK 1: Combined Quota-Fee Optimization
        Uses SAME baseRanking logic as Task 2, but iterates over both quota AND fee ranges.
        """
        
        logger.info(f"🎓 Starting Task 1: Combined Quota-Fee Optimization for Program {program_id}")
        logger.info(f"📊 Quota Range: {min_quota} to {max_quota} (step: {quota_step})")
        logger.info(f"💰 Fee Range: {min_fee:,.0f} to {max_fee:,.0f} TL (step: {fee_step:,.0f})")
        logger.info(f"🎯 Max Ranking Deterioration: {max_ranking_deterioration_pct}%")
        
        # ================================================================================
        # STEP 1: BASELINE PREDICTIONS AND DATA EXTRACTION (SAME AS TASK 2)
        # ================================================================================
        logger.info("\n📋 STEP 1: Extracting baseline data and predictions")
        
        # Use middle values for baseline predictions
        baseline_quota = (min_quota + max_quota) // 2
        baseline_fee = (min_fee + max_fee) / 2
        
        # Get baseline predictions (SAME LOGIC AS TASK 2)
        initial_base_ranking = self.predict_base_ranking_simple(program_id, prediction_year, baseline_fee)
        initial_top_ranking = self.predict_top_ranking_simple(program_id, prediction_year, baseline_fee)
        
        # Extract last year's occupied slots from historical data (SAME AS TASK 2)
        base_input = self.get_base_input_vectors([program_id], prediction_year, 'occupied_slot_v1')
        if base_input.empty or 'lag_occupiedSlots' not in base_input.columns:
            raise ValueError(f"Cannot find lag_occupiedSlots for program {program_id}")
        
        lag_occupied_slots = int(base_input['lag_occupiedSlots'].iloc[0])
        
        # Get baseline occupied prediction
        baseline_occupied_predictions = self.predict_occupied_slots_individual(program_id, prediction_year, baseline_quota, baseline_fee)
        baseline_weights = self.calculate_tier_based_weights(initial_base_ranking, 0, 1)
        baseline_occupied_avg = sum(w * p for w, p in zip(baseline_weights, baseline_occupied_predictions))
        
        # Calculate baseline rank change per student (SAME AS TASK 2)
        if baseline_occupied_avg > 0:
            base_rank_change_per_student = (initial_base_ranking - initial_top_ranking) / baseline_occupied_avg
        else:
            base_rank_change_per_student = 0
        
        # Set ranking constraints
        max_allowed_ranking = min(
            ranking_threshold,
            initial_base_ranking * (1 + max_ranking_deterioration_pct / 100)
        )
        
        # Calculate search space
        quota_range = list(range(min_quota, max_quota + quota_step, quota_step))
        fee_range = list(np.arange(min_fee, max_fee + fee_step, fee_step))
        total_combinations = len(quota_range) * len(fee_range)
        
        logger.info(f"📈 Last Year's Occupied Slots: {lag_occupied_slots}")
        logger.info(f"🏆 Initial Base Ranking: {initial_base_ranking:,.0f}")
        logger.info(f"📊 Base Rank Change per Student: {base_rank_change_per_student:.2f}")
        logger.info(f"🎯 Maximum Allowed Ranking: {max_allowed_ranking:,.0f}")
        logger.info(f"📊 Search Space: {len(quota_range)} quotas × {len(fee_range)} fees = {total_combinations} combinations")
        
        # ================================================================================
        # STEP 2: COMBINED OPTIMIZATION LOOP (ADAPTED FROM TASK 2)
        # ================================================================================
        logger.info("\n🔄 STEP 2: Starting combined quota-fee optimization")
        
        results = []
        best_revenue = 0
        best_combination = None
        quota_iteration = 0
        
        for quota in quota_range:
            quota_iteration += 1
            logger.info(f"\n🔹 QUOTA {quota_iteration}/{len(quota_range)}: Testing Quota {quota}")
            
            # Determine scenario for this quota (SAME LOGIC AS TASK 2)
            if quota > lag_occupied_slots:
                quota_scenario = "EXPANSION"
                logger.info(f"📈 EXPANSION: Quota ({quota}) > lag_occupied ({lag_occupied_slots})")
            elif quota < lag_occupied_slots:
                quota_scenario = "CONTRACTION"
                logger.info(f"📉 CONTRACTION: Quota ({quota}) < lag_occupied ({lag_occupied_slots})")
            else:
                quota_scenario = "MIXED"
                logger.info(f"🔄 MIXED: Quota ({quota}) = lag_occupied ({lag_occupied_slots})")
            
            # Initialize tracking variables for this quota (SAME AS TASK 2)
            expansion_iterations = 0
            contraction_iterations = 0
            mixed_improve_iterations = 0
            mixed_worsen_iterations = 0
            mixed_crossover_point = None
            current_base_ranking = initial_base_ranking
            
            quota_results = []
            consecutive_quota_violations = 0
            
            for fee_iteration, fee in enumerate(fee_range):
                combination_id = f"Q{quota}_F{fee:,.0f}"
                logger.info(f"  💰 Fee {fee_iteration + 1}/{len(fee_range)}: {fee:,.0f} TL")
                
                try:
                    # Calculate model weights for this combination
                    weights = self.calculate_tier_based_weights(current_base_ranking, fee_iteration, len(fee_range))
                    
                    # Get predictions for this quota-fee combination
                    occupied_predictions = self.predict_occupied_slots_individual(program_id, prediction_year, quota, fee)
                    weighted_occupied = sum(w * p for w, p in zip(weights, occupied_predictions))
                    rounded_occupied = round(weighted_occupied)
                    
                    # ========================================================================
                    # APPLY SAME RANKING LOGIC AS TASK 2 FOR EACH COMBINATION
                    # ========================================================================
                    
                    if quota_scenario == "EXPANSION":
                        # EXPANSION LOGIC (SAME AS TASK 2)
                        if rounded_occupied > lag_occupied_slots:
                            expansion_iterations += 1
                            log_factor = math.log(expansion_iterations + 1)
                            rank_change_magnitude = abs(base_rank_change_per_student) * log_factor
                            stability = compute_ranking_stability(current_base_ranking)
                            adjusted_change = rank_change_magnitude * stability
                            excess_students = rounded_occupied - lag_occupied_slots
                            ranking_deterioration = excess_students * adjusted_change
                            current_base_ranking = initial_base_ranking + ranking_deterioration
                        else:
                            current_base_ranking = initial_base_ranking
                            
                    elif quota_scenario == "CONTRACTION":
                        # CONTRACTION LOGIC (SAME AS TASK 2)
                        if rounded_occupied < lag_occupied_slots:
                            contraction_iterations += 1
                            log_factor = math.log(contraction_iterations + 1)
                            rank_change_magnitude = abs(base_rank_change_per_student) / log_factor
                            stability = compute_ranking_stability(current_base_ranking)
                            adjusted_change = rank_change_magnitude * stability
                            deficit_students = lag_occupied_slots - rounded_occupied
                            ranking_deterioration = deficit_students * adjusted_change * 0.3
                            current_base_ranking = initial_base_ranking + ranking_deterioration
                        else:
                            current_base_ranking = initial_base_ranking
                            
                    else:  # MIXED scenario (SAME AS TASK 2)
                        if rounded_occupied <= lag_occupied_slots:
                            # PHASE 1: ranking improves
                            mixed_improve_iterations += 1
                            log_factor = math.log(mixed_improve_iterations + 1)
                            rank_change_magnitude = abs(base_rank_change_per_student) / log_factor
                            stability = compute_ranking_stability(current_base_ranking)
                            adjusted_change = rank_change_magnitude * stability
                            selectivity_increase = lag_occupied_slots - rounded_occupied
                            ranking_improvement = selectivity_increase * adjusted_change * 0.4
                            current_base_ranking = max(0, initial_base_ranking - ranking_improvement)
                        else:
                            # PHASE 2: ranking worsens
                            if mixed_crossover_point is None:
                                mixed_crossover_point = quota
                            mixed_worsen_iterations += 1
                            log_factor = math.log(mixed_worsen_iterations + 1)
                            rank_change_magnitude = abs(base_rank_change_per_student) / log_factor
                            stability = compute_ranking_stability(current_base_ranking)
                            adjusted_change = rank_change_magnitude * stability
                            excess_students = rounded_occupied - lag_occupied_slots
                            ranking_deterioration = excess_students * adjusted_change * 0.5
                            current_base_ranking = initial_base_ranking + ranking_deterioration
                    
                    # Calculate metrics
                    quota_filled = rounded_occupied >= quota
                    actual_occupied = min(quota, rounded_occupied)
                    revenue = actual_occupied * fee
                    ranking_violation = current_base_ranking > max_allowed_ranking
                    ranking_deterioration_pct = ((current_base_ranking - initial_base_ranking) / initial_base_ranking) * 100
                    
                    logger.info(f"    👥 Occupied: {rounded_occupied}, Revenue: {revenue:,.0f} TL, Ranking: {current_base_ranking:,.0f}")
                    
                    # Store result
                    result = {
                        'quota': quota,
                        'fee': fee,
                        'combination_id': combination_id,
                        'quota_scenario': quota_scenario,
                        'weights': weights,
                        'individual_predictions': occupied_predictions,
                        'weighted_occupied': weighted_occupied,
                        'rounded_occupied': rounded_occupied,
                        'actual_occupied': actual_occupied,
                        'quota_filled': quota_filled,
                        'current_base_ranking': current_base_ranking,
                        'ranking_deterioration_pct': ranking_deterioration_pct,
                        'ranking_violation': ranking_violation,
                        'revenue': revenue,
                        'lag_occupied_slots': lag_occupied_slots,
                        'expansion_iterations': expansion_iterations,
                        'contraction_iterations': contraction_iterations,
                        'mixed_improve_iterations': mixed_improve_iterations,
                        'mixed_worsen_iterations': mixed_worsen_iterations,
                        'crossover_point': mixed_crossover_point
                    }
                    
                    results.append(result)
                    quota_results.append(result)
                    
                    # Track best solutions
                    if not ranking_violation:
                        consecutive_quota_violations = 0
                        if revenue > best_revenue:
                            best_revenue = revenue
                            best_combination = result
                            logger.info(f"    🏆 NEW GLOBAL BEST: {combination_id}, Revenue {revenue:,.0f} TL")
                    else:
                        consecutive_quota_violations += 1
                    
                except Exception as e:
                    logger.warning(f"    ❌ Error with combination {combination_id}: {e}")
                    continue
                
                # SAME DYNAMIC STOPPING CONDITIONS AS TASK 2 (adapted for fee loop)
                if consecutive_quota_violations >= 5:
                    logger.info(f"  🛑 STOPPING FEES for quota {quota}: too many violations")
                    break
                
                if quota_filled and not ranking_violation and revenue > best_revenue * 0.8:
                    logger.info(f"  ✅ CONTINUING FEES: Good solution found")
                    continue
                
                if len(quota_results) >= 4:
                    recent_occupied = [r['rounded_occupied'] for r in quota_results[-4:]]
                    recent_revenues = [r['revenue'] for r in quota_results[-4:]]
                    
                    if len(set(recent_occupied)) <= 2:
                        revenue_trend = recent_revenues[-1] - recent_revenues[0]
                        if revenue_trend <= 0:
                            logger.info(f"  🛑 STOPPING FEES for quota {quota}: Plateau detected")
                            break
            
            logger.info(f"🔹 Quota {quota} completed: {len(quota_results)} fee combinations tested")
        
        # ================================================================================
        # STEP 3: FIND OPTIMAL SOLUTION (SAME AS TASK 2)
        # ================================================================================
        logger.info("\n🏆 STEP 3: Finding optimal quota-fee combination")
        
        valid_results = [r for r in results if not r['ranking_violation']]
        
        if valid_results:
            optimal = max(valid_results, key=lambda x: x['revenue'])
            logger.info(f"🎯 Optimal Solution: Quota {optimal['quota']}, Fee {optimal['fee']:,.0f} TL, Revenue {optimal['revenue']:,.0f} TL")
        else:
            optimal = min(results, key=lambda x: x['ranking_deterioration_pct'])
            logger.info(f"⚠️ No fully compliant solution found")
        
        return {
            'program_id': program_id,
            'optimization_type': 'task1_combined_optimization',
            'search_space': {
                'quota_range': {'min': min_quota, 'max': max_quota, 'step': quota_step},
                'fee_range': {'min': min_fee, 'max': max_fee, 'step': fee_step},
                'total_combinations': total_combinations
            },
            'constraints': {
                'ranking_threshold': ranking_threshold,
                'max_ranking_deterioration_pct': max_ranking_deterioration_pct,
                'max_allowed_ranking': max_allowed_ranking
            },
            'initial_predictions': {
                'base_ranking': initial_base_ranking,
                'top_ranking': initial_top_ranking,
                'occupied_slots': baseline_occupied_avg,
                'rank_change_per_student': base_rank_change_per_student
            },
            'lag_occupied_slots': lag_occupied_slots,
            'optimization_results': results,
            'optimal_solution': {
                'quota': optimal['quota'],
                'fee': optimal['fee'],
                'occupied_slots': optimal['actual_occupied'],
                'base_ranking': optimal['current_base_ranking'],
                'revenue': optimal['revenue'],
                'quota_filled': optimal['quota_filled'],
                'ranking_violation': optimal['ranking_violation'],
                'weights_used': optimal['weights']
            },
            'improvements': {
                'revenue_increase': optimal['revenue'] - (baseline_occupied_avg * min_fee),
                'ranking_change': optimal['current_base_ranking'] - initial_base_ranking,
                'quota_change': optimal['quota'] - lag_occupied_slots
            },
            'performance': {
                'total_combinations_tested': len(results),
                'valid_solutions': len(valid_results),
                'completion_rate': len(results) / total_combinations
            }
        }

    def optimize_fee_with_fixed_quota(self, 
                                     program_id: int,
                                     fixed_quota: int,
                                     min_fee: float,
                                     max_fee: float,
                                     fee_step: float = 5000,
                                     ranking_threshold: float = 500000,
                                     max_ranking_deterioration_pct: float = 20,
                                     prediction_year: int = 2025) -> Dict:
        """
        TASK 3: Optimize fee for a fixed quota using the same baseRanking logic.
        """
        
        logger.info(f"🎓 Starting Fixed Quota Fee Optimization for Program {program_id}")
        logger.info(f"📊 Fixed Quota: {fixed_quota} students")
        logger.info(f"💰 Fee Range: {min_fee:,.0f} to {max_fee:,.0f} TL")
        
        # ================================================================================
        # STEP 1: BASELINE PREDICTIONS (SAME LOGIC AS QUOTA OPTIMIZATION)
        # ================================================================================
        logger.info("\n📋 STEP 1: Establishing baseline predictions")
        
        # PREDICT ONCE: Get baseline predictions (FIXED LOGIC)
        baseline_fee = (min_fee + max_fee) / 2
        initial_base_ranking = self.predict_base_ranking_simple(program_id, prediction_year, baseline_fee)
        initial_top_ranking = self.predict_top_ranking_simple(program_id, prediction_year, baseline_fee)
        initial_occupied_predictions = self.predict_occupied_slots_individual(program_id, prediction_year, fixed_quota, baseline_fee)
        
        # Extract last year's data
        base_input = self.get_base_input_vectors([program_id], prediction_year, 'occupied_slot_v1')
        if base_input.empty or 'lag_occupiedSlots' not in base_input.columns:
            raise ValueError(f"Cannot find lag_occupiedSlots for program {program_id}")
        
        lag_occupied_slots = int(base_input['lag_occupiedSlots'].iloc[0])
        
        # Calculate baseline weighted prediction
        baseline_weights = self.calculate_tier_based_weights(initial_base_ranking, 0, 1)
        baseline_occupied_avg = sum(w * p for w, p in zip(baseline_weights, initial_occupied_predictions))
        
        # CALCULATE ONCE: base_rank_change_per_student (FIXED LOGIC)
        if baseline_occupied_avg > 0:
            base_rank_change_per_student = (initial_base_ranking - initial_top_ranking) / baseline_occupied_avg
        else:
            base_rank_change_per_student = 0
        
        # Set ranking constraints
        max_allowed_ranking = min(
            ranking_threshold,
            initial_base_ranking * (1 + max_ranking_deterioration_pct / 100)
        )
        
        logger.info(f"📈 Last Year Occupied: {lag_occupied_slots}")
        logger.info(f"🏆 Initial Base Ranking: {initial_base_ranking:,.0f}")
        logger.info(f"📊 Base Rank Change per Student: {base_rank_change_per_student:.2f}")
        logger.info(f"🎯 Maximum Allowed Ranking: {max_allowed_ranking:,.0f}")
        
        # ================================================================================
        # STEP 2: DETERMINE QUOTA SCENARIO (SAME LOGIC)
        # ================================================================================
        logger.info("\n🔍 STEP 2: Analyzing quota-demand relationship")
        
        if fixed_quota > lag_occupied_slots:
            quota_scenario = "EXPANSION"
            scenario_counter = 0
            logger.info(f"📈 EXPANSION: Fixed quota ({fixed_quota}) > last year ({lag_occupied_slots})")
        elif fixed_quota < lag_occupied_slots:
            quota_scenario = "CONTRACTION"
            scenario_counter = 0
            logger.info(f"📉 CONTRACTION: Fixed quota ({fixed_quota}) < last year ({lag_occupied_slots})")
        else:
            quota_scenario = "STABLE"
            scenario_counter = 0
            logger.info(f"➡️ STABLE: Fixed quota ({fixed_quota}) = last year ({lag_occupied_slots})")
        
        # ================================================================================
        # STEP 3: FEE OPTIMIZATION LOOP (WITH SAME RANKING LOGIC)
        # ================================================================================
        logger.info("\n🔄 STEP 3: Starting fee optimization iterations")
        
        results = []
        fee_range = np.arange(min_fee, max_fee + fee_step, fee_step)
        total_iterations = len(fee_range)
        
        # Tracking variables for dynamic stopping
        consecutive_same_occupied = 0
        consecutive_ranking_violations = 0
        best_revenue = 0
        current_base_ranking = initial_base_ranking
        
        for iteration, current_fee in enumerate(fee_range):
            logger.info(f"\n--- Fee Iteration {iteration + 1}/{total_iterations}: Testing Fee {current_fee:,.0f} TL ---")
            
            # Calculate model weights for current iteration
            weights = self.calculate_tier_based_weights(current_base_ranking, iteration, total_iterations)
            
            # Get predictions for fixed quota and current fee
            occupied_predictions = self.predict_occupied_slots_individual(program_id, prediction_year, fixed_quota, current_fee)
            weighted_occupied = sum(w * p for w, p in zip(weights, occupied_predictions))
            rounded_occupied = round(weighted_occupied)
            
            # Check quota filling
            quota_filled = rounded_occupied >= fixed_quota
            
            # ========================================================================
            # APPLY SAME RANKING LOGIC BASED ON SCENARIO
            # ========================================================================
            
            if quota_scenario == "EXPANSION":
                if rounded_occupied > lag_occupied_slots:
                    scenario_counter += 1
                    current_base_ranking = self.calculate_ranking_change(
                        "EXPANSION", rounded_occupied, lag_occupied_slots, 
                        base_rank_change_per_student, current_base_ranking, 
                        initial_base_ranking, scenario_counter
                    )
                else:
                    current_base_ranking = initial_base_ranking
                    
            elif quota_scenario == "CONTRACTION":
                if rounded_occupied < lag_occupied_slots:
                    scenario_counter += 1
                    current_base_ranking = self.calculate_ranking_change(
                        "CONTRACTION", rounded_occupied, lag_occupied_slots, 
                        base_rank_change_per_student, current_base_ranking, 
                        initial_base_ranking, scenario_counter
                    )
                else:
                    current_base_ranking = initial_base_ranking
            else:  # STABLE
                current_base_ranking = initial_base_ranking
            
            # Calculate revenue
            actual_occupied = min(fixed_quota, rounded_occupied)
            revenue = actual_occupied * current_fee
            
            # Check ranking constraint
            ranking_violation = current_base_ranking > max_allowed_ranking
            ranking_deterioration_pct = ((current_base_ranking - initial_base_ranking) / initial_base_ranking) * 100
            
            logger.info(f"👥 Predicted Occupied: {rounded_occupied}")
            logger.info(f"🏆 Base Ranking: {current_base_ranking:,.0f}")
            logger.info(f"💵 Revenue: {revenue:,.0f} TL")
            logger.info(f"✅ Quota Filled: {'Yes' if quota_filled else 'No'}")
            logger.info(f"⚠️ Ranking Violation: {'Yes' if ranking_violation else 'No'}")
            
            # Store results
            result = {
                'iteration': iteration,
                'fee': current_fee,
                'fixed_quota': fixed_quota,
                'weights': weights,
                'individual_predictions': occupied_predictions,
                'weighted_occupied': weighted_occupied,
                'rounded_occupied': rounded_occupied,
                'actual_occupied': actual_occupied,
                'quota_filled': quota_filled,
                'base_ranking': current_base_ranking,
                'ranking_deterioration_pct': ranking_deterioration_pct,
                'ranking_violation': ranking_violation,
                'revenue': revenue,
                'quota_scenario': quota_scenario
            }
            results.append(result)
            
            # Update best solution
            if not ranking_violation and revenue > best_revenue:
                best_revenue = revenue
                logger.info(f"🏆 NEW BEST: Fee {current_fee:,.0f} TL, Revenue {revenue:,.0f} TL")
            
            # SAME DYNAMIC STOPPING CONDITIONS
            if ranking_violation:
                consecutive_ranking_violations += 1
                if consecutive_ranking_violations >= 3:
                    logger.info(f"🛑 STOPPING: 3 consecutive ranking violations")
                    break
            else:
                consecutive_ranking_violations = 0
            
            if quota_filled:
                logger.info(f"✅ CONTINUING: Quota filled - checking higher fees")
            elif iteration >= 2:
                if abs(rounded_occupied - results[iteration-1]['rounded_occupied']) <= 1:
                    consecutive_same_occupied += 1
                else:
                    consecutive_same_occupied = 0
                
                max_stable_iterations = min(6, total_iterations // 4)
                if consecutive_same_occupied >= max_stable_iterations:
                    recent_revenues = [results[i]['revenue'] for i in range(max(0, iteration-2), iteration+1)]
                    revenue_improving = len(recent_revenues) >= 3 and recent_revenues[-1] > recent_revenues[0]
                    
                    if not revenue_improving:
                        logger.info(f"🛑 STOPPING: Occupied stable, no revenue improvement")
                        break
        
        # ================================================================================
        # STEP 4: FIND OPTIMAL SOLUTION
        # ================================================================================
        logger.info("\n🏆 STEP 4: Finding optimal fee solution")
        
        valid_results = [r for r in results if not r['ranking_violation']]
        
        if valid_results:
            optimal = max(valid_results, key=lambda x: x['revenue'])
            logger.info(f"🎯 Optimal Solution Found: Fee {optimal['fee']:,.0f} TL, Revenue {optimal['revenue']:,.0f} TL")
        else:
            optimal = min(results, key=lambda x: x['ranking_deterioration_pct'])
            logger.info(f"⚠️ No fully compliant solution found. Least violating: Fee {optimal['fee']:,.0f} TL")
        
        return {
            'program_id': program_id,
            'optimization_type': 'fixed_quota_fee_optimization',
            'fixed_quota': fixed_quota,
            'quota_scenario': quota_scenario,
            'fee_range': {'min': min_fee, 'max': max_fee, 'step': fee_step},
            'constraints': {
                'ranking_threshold': ranking_threshold,
                'max_ranking_deterioration_pct': max_ranking_deterioration_pct,
                'max_allowed_ranking': max_allowed_ranking
            },
            'baseline': {
                'base_ranking': initial_base_ranking,
                'top_ranking': initial_top_ranking,
                'occupied_prediction': baseline_occupied_avg,
                'lag_occupied_slots': lag_occupied_slots,
                'rank_change_per_student': base_rank_change_per_student
            },
            'optimization_results': results,
            'optimal_solution': {
                'fee': optimal['fee'],
                'quota': fixed_quota,
                'occupied_slots': optimal['actual_occupied'],
                'base_ranking': optimal['base_ranking'],
                'revenue': optimal['revenue'],
                'quota_filled': optimal['quota_filled'],
                'ranking_violation': optimal['ranking_violation'],
                'ranking_deterioration_pct': optimal['ranking_deterioration_pct']
            },
            'performance': {
                'total_iterations': len(results),
                'valid_solutions': len(valid_results),
                'best_revenue': optimal['revenue'] if valid_results else 0,
                'revenue_improvement': optimal['revenue'] - (baseline_occupied_avg * min_fee) if valid_results else 0
            }
        }

    def optimize_quota_and_fee_combination(self, 
                                         program_id: int,
                                         min_quota: int,
                                         max_quota: int,
                                         min_fee: float,
                                         max_fee: float,
                                         quota_step: int = 5,
                                         fee_step: float = 10000,
                                         ranking_threshold: float = 500000,
                                         max_ranking_deterioration_pct: float = 25,
                                         prediction_year: int = 2025) -> Dict:
        """
        TASK 1: Optimize both quota and fee combination using the same baseRanking logic.
        """
        
        logger.info(f"🎓 Starting Combined Quota-Fee Optimization for Program {program_id}")
        logger.info(f"📊 Quota Range: {min_quota} to {max_quota}")
        logger.info(f"💰 Fee Range: {min_fee:,.0f} to {max_fee:,.0f} TL")
        
        # ================================================================================
        # STEP 1: BASELINE AND SEARCH SPACE ANALYSIS (SAME LOGIC)
        # ================================================================================
        logger.info("\n📋 STEP 1: Analyzing search space and establishing baseline")
        
        # PREDICT ONCE: Get baseline predictions (FIXED LOGIC)
        baseline_quota = (min_quota + max_quota) // 2
        baseline_fee = (min_fee + max_fee) / 2
        initial_base_ranking = self.predict_base_ranking_simple(program_id, prediction_year, baseline_fee)
        initial_top_ranking = self.predict_top_ranking_simple(program_id, prediction_year, baseline_fee)
        
        # Extract historical data
        base_input = self.get_base_input_vectors([program_id], prediction_year, 'occupied_slot_v1')
        if base_input.empty or 'lag_occupiedSlots' not in base_input.columns:
            raise ValueError(f"Cannot find lag_occupiedSlots for program {program_id}")
        
        lag_occupied_slots = int(base_input['lag_occupiedSlots'].iloc[0])
        
        # Get baseline occupied prediction
        baseline_occupied_predictions = self.predict_occupied_slots_individual(program_id, prediction_year, baseline_quota, baseline_fee)
        baseline_weights = self.calculate_tier_based_weights(initial_base_ranking, 0, 1)
        baseline_occupied_avg = sum(w * p for w, p in zip(baseline_weights, baseline_occupied_predictions))
        
        # CALCULATE ONCE: base_rank_change_per_student (FIXED LOGIC)
        if baseline_occupied_avg > 0:
            base_rank_change_per_student = (initial_base_ranking - initial_top_ranking) / baseline_occupied_avg
        else:
            base_rank_change_per_student = 0
        
        # Set ranking constraints
        max_allowed_ranking = min(
            ranking_threshold,
            initial_base_ranking * (1 + max_ranking_deterioration_pct / 100)
        )
        
        # Calculate search space
        quota_range = list(range(min_quota, max_quota + quota_step, quota_step))
        fee_range = list(np.arange(min_fee, max_fee + fee_step, fee_step))
        total_combinations = len(quota_range) * len(fee_range)
        
        logger.info(f"📈 Initial Base Ranking: {initial_base_ranking:,.0f}")
        logger.info(f"📊 Base Rank Change per Student: {base_rank_change_per_student:.2f}")
        logger.info(f"🎯 Maximum Allowed Ranking: {max_allowed_ranking:,.0f}")
        logger.info(f"📊 Search Space: {len(quota_range)} quotas × {len(fee_range)} fees = {total_combinations} combinations")
        logger.info(f"📈 Last Year Occupied: {lag_occupied_slots}")
        
        # ================================================================================
        # STEP 2: SMART GRID SEARCH WITH SAME RANKING LOGIC
        # ================================================================================
        logger.info("\n🔄 STEP 2: Starting smart grid search optimization")
        
        results = []
        best_revenue = 0
        best_combination = None
        quota_iteration = 0
        
        for quota in quota_range:
            quota_iteration += 1
            logger.info(f"\n🔹 QUOTA {quota_iteration}/{len(quota_range)}: Testing Quota {quota}")
            
            # Determine quota scenario for this quota (SAME LOGIC)
            if quota > lag_occupied_slots:
                quota_scenario = "EXPANSION"
            elif quota < lag_occupied_slots:
                quota_scenario = "CONTRACTION"  
            else:
                quota_scenario = "STABLE"
            
            quota_results = []
            consecutive_quota_violations = 0
            best_quota_revenue = 0
            scenario_counter = 0
            current_base_ranking = initial_base_ranking
            
            for fee_iteration, fee in enumerate(fee_range):
                combination_id = f"Q{quota}_F{fee:,.0f}"
                logger.info(f"  💰 Fee {fee_iteration + 1}/{len(fee_range)}: {fee:,.0f} TL")
                
                try:
                    # Calculate model weights
                    weights = self.calculate_tier_based_weights(current_base_ranking, fee_iteration, len(fee_range))
                    
                    # Get predictions for this combination
                    occupied_predictions = self.predict_occupied_slots_individual(program_id, prediction_year, quota, fee)
                    weighted_occupied = sum(w * p for w, p in zip(weights, occupied_predictions))
                    rounded_occupied = round(weighted_occupied)
                    
                    # ========================================================================
                    # APPLY SAME RANKING LOGIC BASED ON QUOTA SCENARIO
                    # ========================================================================
                    
                    if quota_scenario == "EXPANSION":
                        if rounded_occupied > lag_occupied_slots:
                            scenario_counter += 1
                            current_base_ranking = self.calculate_ranking_change(
                                "EXPANSION", rounded_occupied, lag_occupied_slots, 
                                base_rank_change_per_student, current_base_ranking, 
                                initial_base_ranking, scenario_counter
                            )
                        else:
                            current_base_ranking = initial_base_ranking
                            
                    elif quota_scenario == "CONTRACTION":
                        if rounded_occupied < lag_occupied_slots:
                            scenario_counter += 1
                            current_base_ranking = self.calculate_ranking_change(
                                "CONTRACTION", rounded_occupied, lag_occupied_slots, 
                                base_rank_change_per_student, current_base_ranking, 
                                initial_base_ranking, scenario_counter
                            )
                        else:
                            current_base_ranking = initial_base_ranking
                    else:  # STABLE
                        current_base_ranking = initial_base_ranking
                    
                    # Calculate metrics
                    quota_filled = rounded_occupied >= quota
                    actual_occupied = min(quota, rounded_occupied)
                    revenue = actual_occupied * fee
                    ranking_violation = current_base_ranking > max_allowed_ranking
                    ranking_deterioration_pct = ((current_base_ranking - initial_base_ranking) / initial_base_ranking) * 100
                    
                    # Determine demand scenario
                    if rounded_occupied > lag_occupied_slots:
                        demand_scenario = "expansion"
                    elif rounded_occupied < lag_occupied_slots:
                        demand_scenario = "contraction"
                    else:
                        demand_scenario = "stable"
                    
                    logger.info(f"    👥 Occupied: {rounded_occupied}, Revenue: {revenue:,.0f} TL, Ranking: {current_base_ranking:,.0f}")
                    
                    # Store result
                    result = {
                        'quota': quota,
                        'fee': fee,
                        'combination_id': combination_id,
                        'quota_scenario': quota_scenario,
                        'demand_scenario': demand_scenario,
                        'weights': weights,
                        'individual_predictions': occupied_predictions,
                        'weighted_occupied': weighted_occupied,
                        'rounded_occupied': rounded_occupied,
                        'actual_occupied': actual_occupied,
                        'quota_filled': quota_filled,
                        'base_ranking': current_base_ranking,
                        'ranking_deterioration_pct': ranking_deterioration_pct,
                        'ranking_violation': ranking_violation,
                        'revenue': revenue,
                        'lag_occupied_slots': lag_occupied_slots
                    }
                    
                    results.append(result)
                    quota_results.append(result)
                    
                    # Track best solutions
                    if not ranking_violation:
                        consecutive_quota_violations = 0
                        
                        if revenue > best_revenue:
                            best_revenue = revenue
                            best_combination = result
                            logger.info(f"    🏆 NEW GLOBAL BEST: {combination_id}, Revenue {revenue:,.0f} TL")
                        
                        if revenue > best_quota_revenue:
                            best_quota_revenue = revenue
                    else:
                        consecutive_quota_violations += 1
                    
                except Exception as e:
                    logger.warning(f"    ❌ Error with combination {combination_id}: {e}")
                    continue
                
                # SAME DYNAMIC STOPPING CONDITIONS FOR FEE ITERATIONS
                if consecutive_quota_violations >= 5:
                    logger.info(f"  🛑 STOPPING FEES for quota {quota}: too many violations")
                    break
                
                if quota_filled and not ranking_violation and revenue > best_revenue * 0.8:
                    logger.info(f"  ✅ CONTINUING FEES: Good solution found")
                    continue
                
                if len(quota_results) >= 4:
                    recent_occupied = [r['rounded_occupied'] for r in quota_results[-4:]]
                    recent_revenues = [r['revenue'] for r in quota_results[-4:]]
                    
                    if len(set(recent_occupied)) <= 2:
                        revenue_trend = recent_revenues[-1] - recent_revenues[0]
                        if revenue_trend <= 0:
                            logger.info(f"  🛑 STOPPING FEES for quota {quota}: Plateau detected")
                            break
            
            # SAME QUOTA-LEVEL STOPPING CONDITIONS
            logger.info(f"🔹 Quota {quota} completed: {len(quota_results)} fee combinations tested")
            
            if best_quota_revenue == 0:
                logger.info(f"⚠️ No feasible solutions for quota {quota}")
        
        # ================================================================================
        # STEP 3: FIND OPTIMAL SOLUTION
        # ================================================================================
        logger.info("\n🏆 STEP 3: Finding optimal quota-fee combination")
        
        valid_results = [r for r in results if not r['ranking_violation']]
        
        if valid_results:
            optimal = max(valid_results, key=lambda x: x['revenue'])
            logger.info(f"🎯 Optimal Solution: Quota {optimal['quota']}, Fee {optimal['fee']:,.0f} TL, Revenue {optimal['revenue']:,.0f} TL")
        else:
            optimal = min(results, key=lambda x: x['ranking_deterioration_pct'])
            logger.info(f"⚠️ No fully compliant solution found. Least violating: Quota {optimal['quota']}, Fee {optimal['fee']:,.0f} TL")
        
        return {
            'program_id': program_id,
            'optimization_type': 'combined_quota_fee_optimization',
            'search_space': {
                'quota_range': {'min': min_quota, 'max': max_quota, 'step': quota_step},
                'fee_range': {'min': min_fee, 'max': max_fee, 'step': fee_step},
                'total_combinations': total_combinations
            },
            'constraints': {
                'ranking_threshold': ranking_threshold,
                'max_ranking_deterioration_pct': max_ranking_deterioration_pct,
                'max_allowed_ranking': max_allowed_ranking
            },
            'baseline': {
                'base_ranking': initial_base_ranking,
                'top_ranking': initial_top_ranking,
                'lag_occupied_slots': lag_occupied_slots,
                'rank_change_per_student': base_rank_change_per_student
            },
            'optimization_results': results,
            'optimal_solution': {
                'quota': optimal['quota'],
                'fee': optimal['fee'],
                'occupied_slots': optimal['actual_occupied'],
                'base_ranking': optimal['base_ranking'],
                'revenue': optimal['revenue'],
                'quota_filled': optimal['quota_filled'],
                'ranking_violation': optimal['ranking_violation'],
                'ranking_deterioration_pct': optimal['ranking_deterioration_pct'],
                'demand_scenario': optimal['demand_scenario']
            },
            'performance': {
                'total_combinations_tested': len(results),
                'valid_solutions': len(valid_results),
                'completion_rate': len(results) / total_combinations
            }
        }

def main():
    """Main function with all three optimization types."""
    
    # Configuration
    PROJECT_ID = "unioptima-461722"
    DATASET_ID = "university_db"
    SERVICE_ACCOUNT_PATH = "service-account-key.json"
    
    # Initialize advanced optimizer
    optimizer = AdvancedDynamicRevenueOptimizer(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        service_account_path=SERVICE_ACCOUNT_PATH
    )
    
    print("🎓 Advanced Dynamic Revenue Optimizer")
    print("=" * 60)
    print("Choose optimization type:")
    print("1. Quota Optimization (Fixed Fee)")
    print("2. Fee Optimization (Fixed Quota) - TASK 3")
    print("3. Combined Quota-Fee Optimization - TASK 1")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    program_id = int(input("Enter program ID: "))
    
    try:
        if choice == "1":
            # Original quota optimization
            min_quota = int(input("Enter minimum quota to test: "))
            max_quota = int(input("Enter maximum quota to test: "))
            current_fee = float(input("Enter current fee (TL): "))
            threshold = float(input("Enter ranking threshold (default 500000): ") or 500000)
            
            results = optimizer.dynamic_optimize_revenue(
                program_id=program_id,
                current_fee=current_fee,
                min_quota=min_quota,
                max_quota=max_quota,
                ranking_threshold=threshold
            )
            
            display_quota_results(results, min_quota, max_quota, current_fee)
            
        elif choice == "2":
            # TASK 3: Fixed quota fee optimization
            fixed_quota = int(input("Enter fixed quota (kontenjan): "))
            min_fee = float(input("Enter minimum fee to test (TL): "))
            max_fee = float(input("Enter maximum fee to test (TL): "))
            fee_step = float(input("Enter fee step (default 5000 TL): ") or 5000)
            max_deterioration = float(input("Enter max ranking deterioration % (default 20): ") or 20)
            
            results = optimizer.optimize_fee_with_fixed_quota(
                program_id=program_id,
                fixed_quota=fixed_quota,
                min_fee=min_fee,
                max_fee=max_fee,
                fee_step=fee_step,
                max_ranking_deterioration_pct=max_deterioration
            )
            
            display_fee_optimization_results(results)
            
        elif choice == "3":
            # TASK 1: Combined quota-fee optimization
            min_quota = int(input("Enter minimum quota: "))
            max_quota = int(input("Enter maximum quota: "))
            min_fee = float(input("Enter minimum fee (TL): "))
            max_fee = float(input("Enter maximum fee (TL): "))
            quota_step = int(input("Enter quota step (default 5): ") or 5)
            fee_step = float(input("Enter fee step (default 10000 TL): ") or 10000)
            max_deterioration = float(input("Enter max ranking deterioration % (default 25): ") or 25)
            
            results = optimizer.optimize_quota_and_fee_combination(
                program_id=program_id,
                min_quota=min_quota,
                max_quota=max_quota,
                min_fee=min_fee,
                max_fee=max_fee,
                quota_step=quota_step,
                fee_step=fee_step,
                max_ranking_deterioration_pct=max_deterioration
            )
            
            display_combined_optimization_results(results)
            
        else:
            print("❌ Invalid choice!")
            return
            
    except Exception as e:
        logger.error(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()

def display_quota_results(results, min_quota, max_quota, current_fee):
    """Display results for quota optimization."""
    print("\n" + "="*120)
    print("🏆 DYNAMIC QUOTA OPTIMIZATION RESULTS")
    print("="*120)
    
    # Display all the results...
    print(f"\n📋 OPTIMIZATION CONTEXT:")
    print(f"  🎯 Program ID: {results['program_id']}")
    print(f"  📊 Scenario: {results['scenario']}")
    print(f"  📈 Last Year's Occupied: {results['lag_occupied_slots']} students")
    print(f"  🔢 Quota Range Tested: {min_quota} to {max_quota}")
    print(f"  💰 Fee: {current_fee:,.0f} TL")

def display_fee_optimization_results(results):
    """Display results for fixed quota fee optimization (TASK 3)."""
    print("\n" + "="*120)
    print("💰 FIXED QUOTA FEE OPTIMIZATION RESULTS (TASK 3)")
    print("="*120)
    
    # Display comprehensive results for Task 3...
    print(f"\n📋 OPTIMIZATION CONTEXT:")
    print(f"  🎯 Program ID: {results['program_id']}")
    print(f"  📊 Fixed Quota: {results['fixed_quota']} students")
    print(f"  📈 Quota Scenario: {results['quota_scenario']}")

def display_combined_optimization_results(results):
    """Display results for combined quota-fee optimization (TASK 1)."""
    print("\n" + "="*120)
    print("🚀 COMBINED QUOTA-FEE OPTIMIZATION RESULTS (TASK 1)")
    print("="*120)
    
    # Display comprehensive results for Task 1...
    print(f"\n📋 OPTIMIZATION CONTEXT:")
    print(f"  🎯 Program ID: {results['program_id']}")
    print(f"  📊 Search Space: {results['search_space']['total_combinations']} combinations")

if __name__ == "__main__":
    main()