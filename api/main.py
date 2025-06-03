from fastapi import FastAPI
from rl_agent.registry import RewardRegistry

app = FastAPI(
    title="DeepFlyer RL API",
    version="0.1.0"
)

@app.get("/api/rewards/list")
def list_rewards():
    """
    Return the list of registered reward presets (id and label).
    """
    return RewardRegistry.list_presets()
