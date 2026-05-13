import logging
import threading
import mlflow
from app.config import settings

logger = logging.getLogger(__name__)

# Flags to track MLflow state
_mlflow_enabled = True
_mlflow_initialized = False
_lock = threading.Lock()

def initialize_mlflow() -> None:
    """Initializes MLflow tracking URI and Experiment."""
    global _mlflow_enabled, _mlflow_initialized
    
    with _lock:
        if _mlflow_initialized:
            return

    def _init():
        global _mlflow_enabled, _mlflow_initialized
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)
            _mlflow_initialized = True
            logger.info("MLflow initialized successfully at %s", settings.MLFLOW_TRACKING_URI)
        except Exception as exc:
            logger.warning("MLflow initialization failed: %s. Continuing without tracking.", exc)
            _mlflow_enabled = False

    # Threading setup with a 5-second timeout to prevent startup hangs
    init_thread = threading.Thread(target=_init, daemon=True)
    init_thread.start()
    init_thread.join(timeout=5.0)

    if init_thread.is_alive():
        logger.warning("MLflow initialization timed out. Tracking disabled.")
        _mlflow_enabled = False

def log_evaluation(run_name: str, query: str, metrics: dict) -> None:
    """Logs parameters and metrics to MLflow."""
    if not _mlflow_enabled or not _mlflow_initialized:
        return

    try:
        # Start MLflow run
        with mlflow.start_run(run_name=run_name, nested=True):
            # Log the query as a parameter
            mlflow.log_param("query_text", query)
            
            # Separate numeric metrics and non-numeric metadata
            numeric_metrics = {}
            params_metadata = {}
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    numeric_metrics[key] = float(value)
                else:
                    params_metadata[key] = str(value)
            
            # Log numeric values as actual metrics
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics)
            
            # Log strings (like 'consistency') as params
            if params_metadata:
                mlflow.log_params(params_metadata)
                
            # Full JSON dump for detailed review
            mlflow.log_dict(metrics, "evaluation_details.json")
            
    except Exception as exc:
        logger.warning("MLflow logging skipped due to error: %s", exc)