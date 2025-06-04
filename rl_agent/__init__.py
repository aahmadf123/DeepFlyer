from rl_agent.ppo import PPOAgent
from rl_agent.sac import SACAgent
from rl_agent.td3 import TD3Agent
from rl_agent.trpo import TRPOAgent

ALGO_MAP = {
    'ppo': PPOAgent,
    'sac': SACAgent,
    'td3': TD3Agent,
    'trpo': TRPOAgent,
}
