from .trend import compute_expert_scores as trend_scores
from .momentum import compute_expert_scores as momentum_scores
from .mean_reversion import compute_expert_scores as mean_reversion_scores
from .volatility import compute_expert_scores as volatility_scores
from .value import compute_expert_scores as value_scores
from .news_catalyst import compute_expert_scores as news_catalyst_scores

EXPERT_REGISTRY = {
    "trend": trend_scores,
    "momentum": momentum_scores,
    "mean_reversion": mean_reversion_scores,
    "volatility": volatility_scores,
    "value": value_scores,
    "news_catalyst": news_catalyst_scores,
}
