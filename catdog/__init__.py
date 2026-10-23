"""Cat, Dog, or Neither — an open-set pet image classifier.

Package layout
--------------
config       Single source of truth for labels, paths and hyper-parameters.
data/        Dataset sources (Oxford-IIIT Pet, Caltech-101, Imagenette, your own
             photos) and the tf.data pipeline that mixes and balances them.
modeling/    Model definition (pretrained backbone + 3-way head) and training.
inference/   Robust image loading and the Predictor used by the CLI, the
             notebook and the web API.
evaluation/  Confusion matrix, precision/recall and the threshold study that
             justifies the "neither" strategy.
server/      FastAPI application that serves the web interface.
"""

from catdog.config import LABELS

__all__ = ["LABELS"]
__version__ = "1.0.0"
