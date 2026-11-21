"""fairml - an end-to-end, bias-aware MLOps project.

NOTE: keep this file free of heavy imports (torch, mlflow, shap) so the slim
serving image can import `fairml.inference` without them.
"""

__version__ = "1.0.0"
