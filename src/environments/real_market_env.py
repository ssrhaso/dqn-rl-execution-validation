"""
REAL MARKET DATA ENVIRONMENT FOR VALIDATION (USING ALPACA API)

Gym Env that replays real 1-minute OHLCV (open, high, low, close, volume) bar data from Alpaca API for validating the RL Agent performance in real market conditions
"""

import gymnasium as gym
from gymnasium 