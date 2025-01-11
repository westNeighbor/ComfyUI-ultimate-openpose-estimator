from .openpose_estimator_nodes import OpenposeEstimatorNode

NODE_CLASS_MAPPINGS = {
    "OpenposeEstimatorNode": OpenposeEstimatorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenposeEstimatorNode": "Openpose Estimator Node",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
