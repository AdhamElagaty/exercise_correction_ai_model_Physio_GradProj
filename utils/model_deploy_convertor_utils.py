import json
import numpy as np
import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class ModelDeployConverterUtils:

    @staticmethod
    def convert_model_pickle_to_onnx(pickle_model_dir: str, onnx_model_dir: str, name: str, target_opset: int = 12):
        pickle_model_path = os.path.join(pickle_model_dir, f"{name}.pkl")
        onnx_model_path = os.path.join(onnx_model_dir, f"{name}.onnx")

        logger.info(f"Starting conversion for model '{name}': {pickle_model_path} -> {onnx_model_path}")

        if not os.path.exists(pickle_model_path):
            logger.error(f"Pickle model not found at {pickle_model_path}")
            return None

        try:
            os.makedirs(onnx_model_dir, exist_ok=True)
            logger.debug(f"Ensured ONNX output directory exists: {onnx_model_dir}")
        except OSError as e:
            logger.error(f"Could not create ONNX output directory {onnx_model_dir}: {e}")
            return None

        try:
            with open(pickle_model_path, "rb") as f:
                skl_model = pickle.load(f)
            logger.info(f"Successfully loaded pickle model from {pickle_model_path}")
        except (pickle.UnpicklingError, EOFError, FileNotFoundError, Exception) as e:
            logger.error(f"Error loading pickle model {pickle_model_path}: {e}")
            return None

        try:
            if not hasattr(skl_model, 'n_features_in_'):
                if hasattr(skl_model, 'steps'):
                    final_estimator = skl_model.steps[-1][1]
                    if hasattr(final_estimator, 'n_features_in_'):
                        num_features = final_estimator.n_features_in_
                    elif hasattr(final_estimator, 'coef_'):
                         num_features = final_estimator.coef_.shape[-1]
                    elif hasattr(final_estimator, 'feature_importances_'):
                         num_features = len(final_estimator.feature_importances_)
                    else:
                        logger.error(f"Model '{name}' (or its final estimator) does not have a clear way to determine n_features_in_.")
                        return None
                elif hasattr(skl_model, 'coef_'):
                     num_features = skl_model.coef_.shape[-1]
                elif hasattr(skl_model, 'feature_importances_'):
                     num_features = len(skl_model.feature_importances_)
                else:
                    logger.error(f"Model '{name}' does not have 'n_features_in_' attribute or common fallbacks (coef_, feature_importances_). Cannot determine input shape.")
                    return None
            else:
                num_features = skl_model.n_features_in_
            
            logger.info(f"Model for '{name}' expects {num_features} input features.")
        except AttributeError as e:
            logger.error(f"Could not determine number of input features for model '{name}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error determining number of input features for '{name}': {e}")
            return None

        initial_type = [('float_input', FloatTensorType([None, num_features]))]

        try:
            logger.info(f"Attempting to convert model '{name}' to ONNX with target_opset={target_opset}.")
            onnx_model = convert_sklearn(skl_model, initial_types=initial_type, target_opset=target_opset)
            logger.info(f"Successfully converted model '{name}' to ONNX format.")
        except Exception as e:
            logger.error(f"Error converting model '{name}' to ONNX: {e}")
            return None

        try:
            with open(onnx_model_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            logger.info(f"Successfully saved ONNX model to {onnx_model_path}")
            return onnx_model_path
        except IOError as e:
            logger.error(f"Error saving ONNX model to {onnx_model_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error saving ONNX model {onnx_model_path}: {e}")
            return None
    
    @staticmethod
    def convert_scaler_pickle_to_json(scaler_pickle_path: str, json_output_path: str):
        scaler_name_for_log = os.path.basename(scaler_pickle_path)
        logger.info(f"Starting scaler to JSON conversion for '{scaler_name_for_log}': {scaler_pickle_path} -> {json_output_path}")

        if not os.path.exists(scaler_pickle_path):
            logger.error(f"Scaler pickle file not found at {scaler_pickle_path}")
            return None

        json_output_dir = os.path.dirname(json_output_path)
        try:
            if json_output_dir:
                os.makedirs(json_output_dir, exist_ok=True)
            logger.debug(f"Ensured JSON output directory exists: {json_output_dir or '.'}")
        except OSError as e:
            logger.error(f"Could not create JSON output directory {json_output_dir}: {e}")
            return None

        try:
            with open(scaler_pickle_path, "rb") as f:
                scaler = pickle.load(f)
            logger.info(f"Successfully loaded scaler from {scaler_pickle_path}")
        except (pickle.UnpicklingError, EOFError, FileNotFoundError, Exception) as e:
            logger.error(f"Error loading scaler pickle {scaler_pickle_path}: {e}")
            return None

        if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
            logger.error(f"Scaler from '{scaler_pickle_path}' does not have 'mean_' and/or 'scale_' attributes. "
                         "This method assumes a StandardScaler-like object.")
            return None
        
        if not isinstance(scaler.mean_, np.ndarray) or not isinstance(scaler.scale_, np.ndarray):
            logger.error(f"Scaler attributes 'mean_' or 'scale_' from '{scaler_pickle_path}' are not numpy arrays.")
            return None

        scaler_params = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        }
        
        if hasattr(scaler, 'n_features_in_'):
            scaler_params["n_features_in"] = int(scaler.n_features_in_)

        try:
            with open(json_output_path, "w") as f_json:
                json.dump(scaler_params, f_json, indent=4)
            logger.info(f"Successfully converted and saved scaler parameters to {json_output_path}")
            logger.debug(f"  Scaler params for '{scaler_name_for_log}': {scaler_params}")
            return json_output_path
        except (IOError, TypeError, Exception) as e: # TypeError for non-serializable data
            logger.error(f"Error saving scaler parameters to JSON {json_output_path}: {e}")
            return None