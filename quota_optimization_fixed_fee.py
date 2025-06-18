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
            v1_weight = 0.40 - (progress * 0.20)  # High but slightly decreases
            v2_weight = 0.05 + (progress * 0.01)  # Minimal quota effect
            v3_weight = 0.15 + (progress * 0.20)  # Base demand grows
            v4_weight = 0.40 - (progress * 0.10)  # Pure demand grows
            
        elif tier == "good":
            # Good programs: High v1, some quota sensitivity
            quota_sensitivity = 1 - progress
            v1_weight = 0.40 + (quota_sensitivity * 0.20)
            v2_weight = 0.05 + (quota_sensitivity * 0.05)
            v3_weight = 0.25 + (progress * 0.15)
            v4_weight = 0.30 + (progress * 0.10)
            
        elif tier == "average":
            # Average programs: Balanced approach
            quota_sensitivity = (1 - progress) * 0.8
            ranking_factor = min((base_ranking - 10000) / 90000, 1.0)
            
            v1_weight = 0.40 + (quota_sensitivity * 0.10)
            v2_weight = 0.10 + (quota_sensitivity * 0.20) + (ranking_factor * 0.10)
            v3_weight = 0.25 + (progress * 0.15)
            v4_weight = 0.25 + (progress * 0.10)
            
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
        
        logger.debug(f"Tier: {tier}, Progress: {progress:.2f}, Weights: v1={weights[0]:.3f}, v2={weights[1]:.3f}, v3={weights[2]:.3f}, v4={weights[3]:.3f}")
        
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
                logger.debug(f"Model v{i+1}: Updated current_quota to {quota}")
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

    def dynamic_optimize_revenue(self, 
                                program_id: int,
                                current_fee: float,
                                min_quota: int,
                                max_quota: int,
                                ranking_threshold: float = 500000,
                                prediction_year: int = 2025) -> Dict:
        """
        Advanced dynamic optimization with three distinct scenarios and separate iteration counters.
        
        CORE LOGIC:
        1. EXPANSION (min_quota > lag_occupied_slots):
           When predicted > lag_occupied → baseRanking INCREASES (worsens)
           
        2. CONTRACTION (max_quota < lag_occupied_slots):
           When predicted < lag_occupied → baseRanking INCREASES (worsens)
           
        3. MIXED (min_quota < lag_occupied_slots < max_quota):
           Phase 1: predicted ≤ lag_occupied → baseRanking DECREASES (improves)
           Phase 2: predicted > lag_occupied → baseRanking INCREASES (worsens)
        
        Each condition uses separate iteration counters for logarithmic effects.
        """
        
        logger.info(f"🚀 Starting Advanced Dynamic Optimization for Program {program_id}")
        logger.info(f"📊 Quota Range: {min_quota} to {max_quota}")
        logger.info(f"💰 Fee: {current_fee:,.0f} TL")
        
        # ================================================================================
        # STEP 1: BASELINE PREDICTIONS AND DATA EXTRACTION
        # ================================================================================
        logger.info("\n📋 STEP 1: Extracting baseline data and predictions")
        
        # Get baseline predictions
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
        
        logger.info(f"📈 Last Year's Occupied Slots: {lag_occupied_slots}")
        logger.info(f"🏆 Initial Base Ranking: {initial_base_ranking:,.0f}")
        logger.info(f"🥇 Initial Top Ranking: {initial_top_ranking:,.0f}")
        logger.info(f"👥 Initial Predicted Occupied: {initial_occupied_avg:.1f}")
        logger.info(f"🎯 Model Predictions: v1={initial_occupied_predictions[0]:.1f}, v2={initial_occupied_predictions[1]:.1f}, v3={initial_occupied_predictions[2]:.1f}, v4={initial_occupied_predictions[3]:.1f}")
        
        # ================================================================================
        # STEP 2: SCENARIO CLASSIFICATION
        # ================================================================================
        logger.info("\n🔍 STEP 2: Determining optimization scenario")
        
        if min_quota > lag_occupied_slots:
            scenario = "EXPANSION"
            logger.info(f"📈 EXPANSION SCENARIO: min_quota ({min_quota}) > lag_occupied ({lag_occupied_slots})")
            logger.info("🎯 Logic: Higher quotas reduce competition → When predicted > lag → ranking worsens")
        elif max_quota < lag_occupied_slots:
            scenario = "CONTRACTION"
            logger.info(f"📉 CONTRACTION SCENARIO: max_quota ({max_quota}) < lag_occupied ({lag_occupied_slots})")
            logger.info("🎯 Logic: Lower quotas signal demand drop → When predicted < lag → ranking worsens")
        else:
            scenario = "MIXED"
            logger.info(f"🔄 MIXED SCENARIO: min_quota ({min_quota}) < lag_occupied ({lag_occupied_slots}) < max_quota ({max_quota})")
            logger.info("🎯 Logic: Phase 1 (≤lag) → improve ranking | Phase 2 (>lag) → worsen ranking")
        
        # ================================================================================
        # STEP 3: INITIALIZE TRACKING VARIABLES
        # ================================================================================
        logger.info("\n⚙️ STEP 3: Initializing scenario-specific counters and variables")
        
        # Calculate baseline rank change per student
        if initial_occupied_avg > 0:
            base_rank_change_per_student = (initial_base_ranking - initial_top_ranking) / initial_occupied_avg
        else:
            base_rank_change_per_student = 0
        
        logger.info(f"📊 Base Rank Change per Student: {base_rank_change_per_student:.2f}")
        
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
            logger.info(f"⚖️ Model Weights: v1={weights[0]:.3f}, v2={weights[1]:.3f}, v3={weights[2]:.3f}, v4={weights[3]:.3f}")
            
            # Get individual model predictions for current quota
            occupied_predictions = self.predict_occupied_slots_individual(program_id, prediction_year, quota, current_fee)
            logger.info(f"🔮 Raw Predictions: v1={occupied_predictions[0]:.1f}, v2={occupied_predictions[1]:.1f}, v3={occupied_predictions[2]:.1f}, v4={occupied_predictions[3]:.1f}")
            
            # Calculate weighted average and round to whole students
            weighted_occupied = sum(w * p for w, p in zip(weights, occupied_predictions))
            rounded_occupied = round(weighted_occupied)
            quota_filled = rounded_occupied >= quota
            
            logger.info(f"👥 Final Prediction: {rounded_occupied} students (weighted: {weighted_occupied:.1f})")
            logger.info(f"✅ Quota Filled: {'Yes' if quota_filled else 'No'}")
            
            # ========================================================================
            # STEP 4A: APPLY SCENARIO-SPECIFIC RANKING LOGIC
            # ========================================================================
            
            if scenario == "EXPANSION":
                # EXPANSION LOGIC: When predicted > lag_occupied → ranking worsens
                if rounded_occupied > lag_occupied_slots:
                    expansion_iterations += 1
                    
                    # Calculate increasing logarithmic effect
                    log_factor = math.log(expansion_iterations * 2)
                    rank_change_magnitude = abs(base_rank_change_per_student) * log_factor
                    
                    # Apply program-specific stability factor
                    stability = compute_ranking_stability(current_base_ranking)
                    adjusted_change = rank_change_magnitude * stability
                    
                    # Calculate ranking deterioration (increase = worse)
                    excess_students = rounded_occupied - lag_occupied_slots
                    ranking_deterioration = excess_students * adjusted_change 
                    current_base_ranking = initial_base_ranking + ranking_deterioration
                    
                    logger.info(f"📈 EXPANSION: predicted ({rounded_occupied}) > lag ({lag_occupied_slots})")
                    logger.info(f"📊 Ranking worsens by {ranking_deterioration:.0f} → New ranking: {current_base_ranking:.0f}")
                    logger.info(f"🔢 Expansion iterations: {expansion_iterations}")
                else:
                    # Predicted ≤ lag_occupied → minimal ranking change
                    current_base_ranking = initial_base_ranking
                    logger.info(f"➡️ EXPANSION: predicted ({rounded_occupied}) ≤ lag ({lag_occupied_slots}) → No ranking change")
                    
            elif scenario == "CONTRACTION":
                # CONTRACTION LOGIC: When predicted < lag_occupied → ranking worsens
                if rounded_occupied < lag_occupied_slots:
                    contraction_iterations += 1
                    
                    # Calculate diminishing logarithmic effect
                    log_factor = math.log(contraction_iterations * 2)
                    rank_change_magnitude = abs(base_rank_change_per_student) / log_factor
                    
                    # Apply program-specific stability factor
                    stability = compute_ranking_stability(current_base_ranking)
                    adjusted_change = rank_change_magnitude * stability
                    
                    # Calculate ranking deterioration due to demand drop
                    deficit_students = lag_occupied_slots - rounded_occupied
                    ranking_deterioration = deficit_students * adjusted_change * 0.3
                    current_base_ranking = initial_base_ranking + ranking_deterioration
                    
                    logger.info(f"📉 CONTRACTION: predicted ({rounded_occupied}) < lag ({lag_occupied_slots})")
                    logger.info(f"📊 Ranking worsens by {ranking_deterioration:.0f} → New ranking: {current_base_ranking:.0f}")
                    logger.info(f"🔢 Contraction iterations: {contraction_iterations}")
                else:
                    # Predicted ≥ lag_occupied → minimal ranking change
                    current_base_ranking = initial_base_ranking
                    logger.info(f"➡️ CONTRACTION: predicted ({rounded_occupied}) ≥ lag ({lag_occupied_slots}) → No ranking change")
                    
            else:  # scenario == "MIXED"
                # MIXED LOGIC: Two distinct phases with crossover detection
                if rounded_occupied <= lag_occupied_slots:
                    # PHASE 1: More selective than last year → ranking improves
                    mixed_improve_iterations += 1
                    
                    # Calculate diminishing logarithmic effect
                    log_factor = math.log(mixed_improve_iterations * 2)
                    rank_change_magnitude = abs(base_rank_change_per_student) / log_factor
                    
                    # Apply program-specific stability factor
                    stability = compute_ranking_stability(current_base_ranking)
                    adjusted_change = rank_change_magnitude * stability
                    
                    # Calculate ranking improvement (decrease = better)
                    selectivity_increase = lag_occupied_slots - rounded_occupied
                    ranking_improvement = selectivity_increase * adjusted_change * 0.4
                    current_base_ranking = max(0, initial_base_ranking - ranking_improvement)
                    
                    logger.info(f"⬇️ MIXED-IMPROVE: predicted ({rounded_occupied}) ≤ lag ({lag_occupied_slots})")
                    logger.info(f"📊 Ranking improves by {ranking_improvement:.0f} → New ranking: {current_base_ranking:.0f}")
                    logger.info(f"🔢 Mixed improve iterations: {mixed_improve_iterations}")
                    
                else:
                    # PHASE 2: Less selective than last year → ranking worsens
                    if mixed_crossover_point is None:
                        mixed_crossover_point = quota
                        logger.info(f"🎯 MIXED CROSSOVER DETECTED at quota {quota}!")
                    
                    mixed_worsen_iterations += 1
                    
                    # Calculate diminishing logarithmic effect
                    log_factor = math.log(mixed_worsen_iterations * 2)
                    rank_change_magnitude = abs(base_rank_change_per_student) * log_factor
                    
                    # Apply program-specific stability factor
                    stability = compute_ranking_stability(current_base_ranking)
                    adjusted_change = rank_change_magnitude * stability
                    
                    # Calculate ranking deterioration
                    excess_students = rounded_occupied - lag_occupied_slots
                    ranking_deterioration = excess_students * adjusted_change * 0.5
                    current_base_ranking = initial_base_ranking + ranking_deterioration
                    
                    logger.info(f"⬆️ MIXED-WORSEN: predicted ({rounded_occupied}) > lag ({lag_occupied_slots})")
                    logger.info(f"📊 Ranking worsens by {ranking_deterioration:.0f} → New ranking: {current_base_ranking:.0f}")
                    logger.info(f"🔢 Mixed worsen iterations: {mixed_worsen_iterations}")
            
            # Calculate revenue for this iteration
            rounded_occupied = min(quota, rounded_occupied)
            revenue = rounded_occupied * current_fee
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
                logger.info(f"🛑 STOPPING (Priority 1): Ranking exceeded threshold ({current_base_ranking:,.0f} > {ranking_threshold:,.0f})")
                break
            elif ranking_deterioration_ratio > 3.0:  # Ranking became 3x worse
                logger.info(f"🛑 STOPPING (Priority 1): Ranking deteriorated too much ({ranking_deterioration_ratio:.2f}x worse)")
                break
            
            # PRIORITY 2: If quota is filled, ALWAYS continue (override other stopping conditions)
            if quota_filled:
                logger.info(f"✅ CONTINUING: Quota filled ({rounded_occupied} ≥ {quota}) - ignoring other stop conditions")
                continue
            
            # PRIORITY 3: Dynamic occupiedSlot stability analysis (only if quota not filled)
            if iteration >= 2:  # Need at least 3 data points for analysis
                # Get recent occupiedSlot history
                recent_occupied = [results[i]['rounded_occupied'] for i in range(max(0, iteration-4), iteration+1)]
                recent_quotas = [results[i]['quota'] for i in range(max(0, iteration-4), iteration+1)]
                
                # Calculate stability metrics
                occupied_changes = [abs(recent_occupied[i] - recent_occupied[i-1]) for i in range(1, len(recent_occupied))]
                quota_changes = [recent_quotas[i] - recent_quotas[i-1] for i in range(1, len(recent_quotas))]
                
                # Stability conditions
                max_stable_iterations = min(8, total_iterations // 3)  # Dynamic based on total iterations
                consecutive_stable = 0
                
                # Count consecutive iterations with same occupiedSlot
                for i in range(len(occupied_changes)-1, -1, -1):
                    if occupied_changes[i] <= 1:  # Allow ±1 student variation
                        consecutive_stable += 1
                    else:
                        break
                
                # Advanced stability analysis
                if len(recent_occupied) >= 5:
                    # Check if we're in a "plateau" pattern
                    unique_occupied = len(set(recent_occupied))
                    quota_range = max(recent_quotas) - min(recent_quotas)
                    
                    # Plateau detection: same occupiedSlot despite increasing quotas
                    if unique_occupied <= 2 and quota_range >= 5:
                        logger.info(f"📊 PLATEAU DETECTED: {unique_occupied} unique occupied values over {quota_range} quota range")
                        
                        # But don't stop immediately - check if we might break through
                        remaining_iterations = total_iterations - iteration - 1
                        if consecutive_stable >= max_stable_iterations and remaining_iterations < 3:
                            logger.info(f"🛑 STOPPING (Priority 3): Plateau confirmed with {consecutive_stable} stable iterations")
                            break
                        else:
                            logger.info(f"⏳ CONTINUING: Plateau detected but checking for potential breakthrough ({remaining_iterations} iterations left)")
                
                # Standard stability check
                elif consecutive_stable >= max_stable_iterations:
                    # Additional check: are we making progress in any metric?
                    recent_revenues = [results[i]['revenue'] for i in range(max(0, iteration-2), iteration+1)]
                    revenue_improving = len(recent_revenues) >= 3 and recent_revenues[-1] > recent_revenues[0]
                    
                    if revenue_improving:
                        logger.info(f"⏳ CONTINUING: {consecutive_stable} stable iterations but revenue improving")
                    else:
                        logger.info(f"🛑 STOPPING (Priority 3): OccupiedSlot stable for {consecutive_stable} iterations (max: {max_stable_iterations})")
                        break
                
                # Log current stability status
                if consecutive_stable > 0:
                    logger.info(f"📊 Stability: {consecutive_stable}/{max_stable_iterations} stable iterations")
            
            # PRIORITY 4: Scenario-specific advanced stopping conditions
            if scenario == "EXPANSION":
                # In expansion, if we've tested many quotas above lag but prediction doesn't increase
                if expansion_iterations >= 8:
                    recent_predictions = [results[i]['rounded_occupied'] for i in range(max(0, iteration-3), iteration+1)]
                    if max(recent_predictions) <= lag_occupied_slots:
                        logger.info(f"🛑 STOPPING (Expansion): {expansion_iterations} expansions tested, predictions not exceeding lag ({max(recent_predictions)} ≤ {lag_occupied_slots})")
                        break
                elif expansion_iterations >= 15:
                    logger.info(f"🛑 STOPPING (Expansion): Maximum expansion iterations reached ({expansion_iterations})")
                    break
                    
            elif scenario == "CONTRACTION":
                # In contraction, if we've tested many quotas below lag but prediction doesn't decrease much
                if contraction_iterations >= 8:
                    recent_predictions = [results[i]['rounded_occupied'] for i in range(max(0, iteration-3), iteration+1)]
                    min_prediction = min(recent_predictions)
                    if lag_occupied_slots - min_prediction < 5:  # Less than 5 student difference
                        logger.info(f"🛑 STOPPING (Contraction): Limited contraction effect detected (min: {min_prediction} vs lag: {lag_occupied_slots})")
                        break
                elif contraction_iterations >= 15:
                    logger.info(f"🛑 STOPPING (Contraction): Maximum contraction iterations reached ({contraction_iterations})")
                    break
                    
            else:  # MIXED scenario
                # In mixed, check if we're stuck in one phase too long
                total_mixed_iterations = mixed_improve_iterations + mixed_worsen_iterations
                if mixed_worsen_iterations >= 12:
                    logger.info(f"🛑 STOPPING (Mixed): Too many deterioration iterations ({mixed_worsen_iterations})")
                    break
                elif total_mixed_iterations >= 20:
                    logger.info(f"🛑 STOPPING (Mixed): Total mixed iterations limit reached ({total_mixed_iterations})")
                    break
                
                # Special case: if we've been in improve phase for long but no crossover
                if mixed_improve_iterations >= 10 and mixed_worsen_iterations == 0:
                    remaining_iterations = total_iterations - iteration - 1
                    if remaining_iterations < 3:
                        logger.info(f"🛑 STOPPING (Mixed): Long improve phase without crossover, few iterations left")
                        break
            
            # PRIORITY 5: Resource optimization (prevent extremely long runs)
            if iteration >= total_iterations * 0.9:  # 90% of planned iterations
                # Check if we're making meaningful progress
                if len(results) >= 10:
                    early_revenue = results[len(results)//2]['revenue']
                    current_revenue = results[-1]['revenue']
                    progress_rate = (current_revenue - early_revenue) / early_revenue if early_revenue > 0 else 0
                    
                    if abs(progress_rate) < 0.02:  # Less than 2% revenue change in latter half
                        logger.info(f"🛑 STOPPING (Priority 5): Minimal progress in recent iterations ({progress_rate:.1%})")
                        break
        
        # ================================================================================
        # STEP 5: FIND OPTIMAL SOLUTION AND COMPILE RESULTS
        # ================================================================================
        logger.info("\n🏆 STEP 5: Finding optimal solution")
        
        # Find the solution that maximizes revenue while maintaining reasonable ranking
        valid_results = [r for r in results if r['rounded_occupied'] > 0]
        if valid_results:
            # Prioritize revenue but penalize excessive ranking deterioration
            def optimization_score(result):
                revenue_score = result['revenue']
                ranking_penalty = max(0, result['current_base_ranking'] - initial_base_ranking) * 10
                return revenue_score - ranking_penalty
            
            optimal = max(valid_results, key=optimization_score)
        else:
            optimal = results[0]
        
        logger.info(f"🎯 Optimal Solution Found:")
        logger.info(f"  📊 Quota: {optimal['quota']}")
        logger.info(f"  👥 Occupied Slots: {optimal['rounded_occupied']}")
        logger.info(f"  🏆 Base Ranking: {optimal['current_base_ranking']:,.0f}")
        logger.info(f"  💰 Revenue: {optimal['revenue']:,.0f} TL")
        
        # Compile comprehensive results
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
                'occupied_slots': optimal['rounded_occupied'],
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

def main():
    """Advanced optimizer usage example with comprehensive output."""
    
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
    
    # Get user input
    print("🎓 Advanced Dynamic Revenue Optimizer")
    print("=" * 50)
    program_id = int(input("Enter program ID: "))
    min_quota = int(input("Enter minimum quota to test: "))
    max_quota = int(input("Enter maximum quota to test: "))
    current_fee = float(input("Enter current fee (TL): "))
    threshold = float(input("Enter ranking threshold (default 500000): ") or 500000)
    
    try:
        # Run optimization
        results = optimizer.dynamic_optimize_revenue(
            program_id=program_id,
            current_fee=current_fee,
            min_quota=min_quota,
            max_quota=max_quota,
            ranking_threshold=threshold
        )

        # Display comprehensive results
        print("\n" + "="*120)
        print("🏆 ADVANCED DYNAMIC REVENUE OPTIMIZATION RESULTS")
        print("="*120)
        
        # Scenario and context
        print(f"\n📋 OPTIMIZATION CONTEXT:")
        print(f"  🎯 Program ID: {program_id}")
        print(f"  📊 Scenario: {results['scenario']}")
        print(f"  📈 Last Year's Occupied: {results['lag_occupied_slots']} students")
        print(f"  🔢 Quota Range Tested: {min_quota} to {max_quota}")
        print(f"  💰 Fee: {current_fee:,.0f} TL")
        
        # Initial baseline
        initial = results['initial_predictions']
        print(f"\n📋 BASELINE PREDICTIONS:")
        print(f"  🏆 Base Ranking: {initial['base_ranking']:,.0f}")
        print(f"  🥇 Top Ranking: {initial['top_ranking']:,.0f}")
        print(f"  👥 Predicted Occupied: {initial['occupied_slots']:.1f} students")
        print(f"  📊 Rank Change/Student: {initial['rank_change_per_student']:.2f}")
        
        # Optimal solution
        optimal = results['optimal_solution']
        print(f"\n🏆 OPTIMAL SOLUTION:")
        print(f"  📊 Optimal Quota: {optimal['quota']} students")
        print(f"  👥 Expected Occupied: {optimal['occupied_slots']} students")
        print(f"  🏆 Final Base Ranking: {optimal['base_ranking']:,.0f}")
        print(f"  💰 Expected Revenue: {optimal['revenue']:,.0f} TL")
        print(f"  ⚖️ Model Weights: v1={optimal['weights_used'][0]:.3f}, v2={optimal['weights_used'][1]:.3f}, v3={optimal['weights_used'][2]:.3f}, v4={optimal['weights_used'][3]:.3f}")
        
        # Performance improvements
        improvements = results['improvements']
        print(f"\n📈 PERFORMANCE IMPROVEMENTS:")
        print(f"  💰 Revenue Increase: {improvements['revenue_increase']:,.0f} TL")
        print(f"  📊 Quota Change: {improvements['quota_change']:+d} students")
        print(f"  🏆 Ranking Change: {improvements['ranking_change']:+,.0f}")
        print(f"  📊 Revenue % Increase: {(improvements['revenue_increase'] / (initial['occupied_slots'] * current_fee) * 100):+.1f}%")
        
        # Scenario-specific statistics
        stats = results['scenario_statistics']
        print(f"\n📊 SCENARIO ANALYSIS:")
        print(f"  🔄 Total Iterations: {stats['total_iterations_run']}")
        
        if results['scenario'] == "EXPANSION":
            print(f"  📈 Expansion Iterations: {stats['expansion_iterations']}")
            print(f"  💡 Strategy: Monitored quota expansion effects on competitiveness")
        elif results['scenario'] == "CONTRACTION":
            print(f"  📉 Contraction Iterations: {stats['contraction_iterations']}")
            print(f"  💡 Strategy: Analyzed demand reduction impact on ranking")
        else:  # MIXED
            print(f"  ⬇️ Improvement Phase: {stats['mixed_improve_iterations']} iterations")
            print(f"  ⬆️ Deterioration Phase: {stats['mixed_worsen_iterations']} iterations")
            if stats['crossover_point']:
                print(f"  🎯 Crossover at Quota: {stats['crossover_point']}")
            print(f"  💡 Strategy: Two-phase optimization with crossover detection")
        
        # Recent iteration details
        print(f"\n🔍 RECENT ITERATION DETAILS:")
        recent_results = results['optimization_results'][-5:]  # Last 5 iterations
        for i, result in enumerate(recent_results, 1):
            print(f"  {len(recent_results)-i+1:2d}. Quota {result['quota']:3d}: "
                  f"occupied={result['rounded_occupied']:3d}, "
                  f"ranking={result['current_base_ranking']:8,.0f}, "
                  f"revenue={result['revenue']:10,.0f} TL, "
                  f"filled={'✅' if result['quota_filled'] else '❌'}")
        
        # Strategic recommendations
        print(f"\n💡 STRATEGIC RECOMMENDATIONS:")
        if improvements['revenue_increase'] > 0:
            print(f"  ✅ Recommend implementing optimal quota of {optimal['quota']} students")
            print(f"  📈 Expected revenue boost: {improvements['revenue_increase']:,.0f} TL")
        else:
            print(f"  ⚠️ Current configuration appears optimal")
            print(f"  🔍 Consider other fee or capacity adjustments")
        
        if abs(improvements['ranking_change']) < initial['base_ranking'] * 0.1:
            print(f"  ✅ Ranking impact is minimal and acceptable")
        else:
            print(f"  ⚠️ Significant ranking change detected - monitor carefully")
        
    except Exception as e:
        logger.error(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()