import numpy as np
import pandas as pd
import json

def evaluate_domain_constraints(synth_df, dataset_name):
    """
    Evaluates dataset-specific strict domain constraints.
    Returns the percentage of rows that VIOLATE the constraint (Lower is better, 0.0 is perfect).
    """
    violations = {}
    
    # Fast-fail for empty data
    if synth_df.empty:
        return {"Constraint_Violation_Rate": np.nan}

    try:
        if dataset_name == 'cern':
            # CONSTRAINT: Relativistic Kinematics
            # M = sqrt(2 * pT1 * pT2 * (cosh(eta1 - eta2) - cos(phi1 - phi2)))
            
            # Look for the required columns (handling the trailing underscores in your schema)
            cern_cols = ['pt1_', 'pt2_', 'eta1_', 'eta2_', 'phi1_', 'phi2_', 'M_']
            
            if all(c in synth_df.columns for c in cern_cols):
                # Safely parse to numeric, forcing garbage/NaNs to 0
                pt1 = pd.to_numeric(synth_df['pt1_'], errors='coerce').fillna(0)
                pt2 = pd.to_numeric(synth_df['pt2_'], errors='coerce').fillna(0)
                eta_diff = pd.to_numeric(synth_df['eta1_'], errors='coerce').fillna(0) - \
                           pd.to_numeric(synth_df['eta2_'], errors='coerce').fillna(0)
                phi_diff = pd.to_numeric(synth_df['phi1_'], errors='coerce').fillna(0) - \
                           pd.to_numeric(synth_df['phi2_'], errors='coerce').fillna(0)
                gen_m = pd.to_numeric(synth_df['M_'], errors='coerce').fillna(0)
                
                # Calculate the theoretical M^2
                calc_m_squared = 2 * pt1 * pt2 * (np.cosh(eta_diff) - np.cos(phi_diff))
                
                # Clip to 0 to prevent NaN from np.sqrt() if the model hallucinated negative pt values
                calc_m_squared = calc_m_squared.clip(lower=0)
                calc_m = np.sqrt(calc_m_squared)
                
                # Check if generated M matches calculated M (allowing a small float tolerance)
                epsilon = 0.05
                violation_mask = (gen_m - calc_m).abs() > epsilon
                
                violations['Kinematic_Violation'] = violation_mask.mean()

        elif dataset_name == 'lob':
            # CONSTRAINT: Financial Order Book Bounds
            # Distances (prices) and Notional Volumes cannot physically fall below zero.
            distance_cols = [c for c in synth_df.columns if 'distance' in c]
            notional_cols = [c for c in synth_df.columns if 'notional' in c]
            
            if distance_cols and notional_cols:
                dist_viol = (synth_df[distance_cols].apply(pd.to_numeric, errors='coerce').fillna(0) < 0).any(axis=1)
                not_viol = (synth_df[notional_cols].apply(pd.to_numeric, errors='coerce').fillna(0) < 0).any(axis=1)
                
                violation_mask = dist_viol | not_viol
                violations['Negative_Volume_Price_Violation'] = violation_mask.mean()

        elif dataset_name == 'bayesian':
            # CONSTRAINT: Strict Deterministic Addition (The Collider Trap)
            # The 'collider' must equal 'root_1' + 'root_2' (+ small noise epsilon)
            if all(c in synth_df.columns for c in ['root_1', 'root_2', 'collider']):
                r1 = pd.to_numeric(synth_df['root_1'], errors='coerce').fillna(0)
                r2 = pd.to_numeric(synth_df['root_2'], errors='coerce').fillna(0)
                collider = pd.to_numeric(synth_df['collider'], errors='coerce').fillna(0)
                
                epsilon = 0.35 
                actual_sum = r1 + r2
                
                violation_mask = abs(collider - actual_sum) > epsilon
                violations['Additive_Logic_Violation'] = violation_mask.mean()

        elif dataset_name == 'olist':
            # CONSTRAINT: Temporal Causality (Time flow)
            # A package cannot be delivered before it is purchased.
            if 'order_purchase_timestamp' in synth_df.columns and 'order_delivered_customer_date' in synth_df.columns:
                
                # Use to_datetime to safely parse both raw strings and float epochs
                # If they are already numeric epochs from CTGAN, we treat them as seconds
                def parse_mixed_dates(series):
                    if pd.api.types.is_numeric_dtype(series):
                        return pd.to_datetime(series, unit='s', errors='coerce')
                    return pd.to_datetime(series, errors='coerce')

                purchase = parse_mixed_dates(synth_df['order_purchase_timestamp'])
                delivery = parse_mixed_dates(synth_df['order_delivered_customer_date'])
                
                # Violation: Delivered before purchased
                violation_mask = (delivery < purchase)
                violations['Temporal_Causality_Violation'] = violation_mask.mean()
                
        elif dataset_name == 'honeypot':
            # CONSTRAINT: JSON Syntactic Parse Rate
            # Models must generate valid, compileable JSON arrays/objects without hallucinating 
            # mismatched quotes or broken brackets.
            if 'request_raw' in synth_df.columns:
                def is_broken_json(val):
                    val = str(val).strip()
                    # Skip empty strings or non-JSON HTTP requests
                    if not val or val == "Missing" or val == "nan" or val=="":
                        return False
                    
                    # If it looks like it is attempting to be a JSON object/array
                    if val.startswith('{') or val.startswith('[') or ('{' in val and '}' in val):
                        try:
                            # If it parses, it's mathematically sound syntax
                            json.loads(val)
                            return False 
                        except json.JSONDecodeError:
                            # The model failed to close a bracket/quote
                            return True
                    return False
                    
                violation_mask = synth_df['request_raw'].apply(is_broken_json)
                violations['JSON_Parse_Violation'] = violation_mask.mean()

        # Aggregate the specific domain violation into a single standardized column for the summary table
        if violations:
            primary_key = list(violations.keys())[0]
            return {"Constraint_Violation_Rate": float(violations[primary_key])}
        else:
            return {"Constraint_Violation_Rate": np.nan}
            
    except Exception as e:
        print(f"  [!] Domain constraint evaluation failed: {e}")
        return {"Constraint_Violation_Rate": np.nan}
    