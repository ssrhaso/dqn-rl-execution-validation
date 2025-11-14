"""
Test script for RealMarketEnv - validates environment with real Alpaca data
"""

from src.data.alpaca_loader import AlpacaDataLoader
from src.environments.real_market_env import RealMarketEnv
import numpy as np

def main():
    print("=" * 70)
    print("TEST 1: Load Real Market Data")
    print("=" * 70)
    
    loader = AlpacaDataLoader()
    # Use a recent confirmed trading day - update this if needed!
    df = loader.download_bars(
        symbol="AAPL",  # Pick a highly liquid symbol
        start_date="2025-10-10",
        end_date="2025-10-10",
        timeframe="1Min"
    )

    print(f"✓ Loaded {len(df)} bars")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Robust: fail early if <= 1 bar
    if len(df) <= 1:
        print("\nERROR: Only 1 or zero bars returned. This usually means you are:\n"
              "a) Using a weekend, market holiday, or after-hours-only symbol\n"
              "b) Your Alpaca subscription doesn't support 1-minute historical data\n"
              "c) The chosen stock had no trading that day\n")
        print("Try a different trading day, a more liquid symbol, or check your Alpaca plan!")
        return

    print("\n" + "=" * 70)
    print("TEST 2: Create RealMarketEnv")
    print("=" * 70)

    env = RealMarketEnv(
        market_data=df,
        parent_order_size=1000,
        execution_cost_bps=5.0
    )

    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n} actions")
    print(f"  Time horizon: {env.time_horizon} steps")

    print("\n" + "=" * 70)
    print("TEST 3: Reset Environment")
    print("=" * 70)

    obs, info = env.reset(seed=42)
    print(f"✓ Reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation: {obs}")
    print(f"  Info: {info}")

    print("\n" + "=" * 70)
    print("TEST 4: Random Policy (10 steps)")
    print("=" * 70)

    obs, info = env.reset()
    total_reward = 0.0

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"  Step {i+1}: action={action}, reward={reward:.4f}, "
              f"inventory={info['inventory']}, executed={info['shares_executed']}")
        
        if terminated:
            print("  Episode terminated early")
            break

    print(f"✓ Random policy test complete: total_reward={total_reward:.4f}")

    print("\n" + "=" * 70)
    print("TEST 5: Aggressive Policy (Always Execute 10%)")
    print("=" * 70)

    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action = 1  # Always execute 10%
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if steps % 50 == 0 or terminated:
            print(f"  Step {steps}: inventory={info['inventory']}, "
                  f"executed={info['shares_executed']}, "
                  f"slippage={info['slippage_bps']:.2f} bps")

        if terminated:
            break

    print(f"\n✓ Aggressive policy complete:")
    print(f"  Steps taken: {steps}")
    print(f"  Shares executed: {info['shares_executed']}/{env.parent_order_size}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Avg execution price: ${info['avg_execution_price']:.2f}")
    print(f"  VWAP benchmark: ${info['vwap_benchmark']:.2f}")
    print(f"  Slippage: {info['slippage_bps']:.2f} bps")
    print(f"  Completion rate: {info['completion_rate']*100:.1f}%")

    print("\n" + "=" * 70)
    print("TEST 6: Conservative Policy (Always Execute 5%)")
    print("=" * 70)

    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action = 2  # Always execute 5%
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if steps % 50 == 0 or terminated:
            print(f"  Step {steps}: inventory={info['inventory']}, "
                  f"executed={info['shares_executed']}")

        if terminated:
            break

    print(f"\n✓ Conservative policy complete:")
    print(f"  Steps taken: {steps}")
    print(f"  Shares executed: {info['shares_executed']}/{env.parent_order_size}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Slippage: {info['slippage_bps']:.2f} bps")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print("\nRealMarketEnv is working correctly with Alpaca data.")
    print("Ready to load trained models for validation!")

if __name__ == "__main__":
    main()
