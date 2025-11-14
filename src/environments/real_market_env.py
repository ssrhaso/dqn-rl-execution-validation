"""
REAL MARKET DATA ENVIRONMENT FOR VALIDATION (USING ALPACA API)

Gym Env that replays real 1-minute OHLCV (open, high, low, close, volume) bar data from Alpaca API for validating the RL Agent performance in real market conditions
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
import logging
logger = logging.getLogger(__name__)


class RealMarketEnv(gym.Env):
    """
    STATE SPACE (8 DIM):
        1. NORM BEST BID (PROXY VWAP)
        2. NORM BEST ASK (PROXY VWAP)
        3. NORM SPREAD
        4. TIME PROGRESS (0 TO 1)
        5. INV REMAINING (0 TO 1)
        6. CURRENT VOLATILITY
        7. EXECUTION PROGRESS (0 TO 1)
        8. NORM AVG EXECUTION PRICE
        
    ACTION SPACE (3 DIM):
        0. WAIT (NO ACTION)
        1. EXECUTE 10% OF REMAINING INV
        2. EXECUTE 5% OF REMAINING INV        
    """
    
    def __init__(
        self,
        market_data : pd.DataFrame,                 # DataFrame with OHLCV bars
        parent_order_size : int = 1000,             # Total shares to execute
        execution_cost_bps : float = 5.0,           # Slippage + fees in basis points (bps)
        spread_factor : float = 0.002               # Synthetic spread as fraction of price (e.g., 0.002 = 0.2% spread)
    ):
        super().__init__() # Initialise the base Gym environment
        
        # VALIDATION - MARKET DATA
        required_columns = ['timestamp','open', 'high', 'low', 'close', 'volume', 'vwap']
        missing_cols = [col for col in required_columns if col not in market_data.columns]
        if missing_cols:
            raise ValueError(f"Market data is missing required columns: {missing_cols}")
        if len(market_data) == 0:
            raise ValueError("Market data is empty.")
        print(f"RealMarketEnv: Loaded market data with {len(market_data)} rows and columns: {market_data.columns.tolist()}")
        
        # STORE MARKET DATA
        self.market_data = market_data.reset_index(drop=True).copy()            # Store DF in env for each index
        self.parent_order_size = parent_order_size                              # Store shares to execute
        self.execution_cost_bps = execution_cost_bps / 10000.0                  # Store bps and convert to decimal
        self.spread_factor = spread_factor                                      # Store spread factor     
        self.time_horizon = len(self.market_data)                               # Total time steps in episode   
        
        
        # OBSERVATION SPACE (8 DIM)
        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (8,),# 8 DIM           
            dtype = np.float32
        )
        # ACTION SPACE (3 DIM)
        self.action_space = spaces.Discrete(3)  # 3 DIM ACTION SPACE
        
        # INITIALISE STATE VARIABLES
        self.current_step = 0                     # Current time step index
        self.inventory = parent_order_size        # Remaining shares to execute
        self.cash_spent = 0.0                     # Total cash spent on executions
        self.shares_executed = 0                  # Total shares executed
        self.execution_prices = []                # List of execution prices for averaging
        self.execution_history = []               # History of executions (step, shares, price)
        self._calculate_volatility()              # Pre-calculate volatility for each bar
        
        logger.info(f"RealMarketEnv initialised: {len(self.market_data)} BARS OF MARKET DATA "
                    f"FOR ORDER SIZE {self.parent_order_size}, TIME HORIZON {self.time_horizon}.")
        
    # VOLATILITY CALCULATION
    def _calculate_volatility(self, window : int = 10):
        """ ROLLING VOLATILITY (10 BARS)"""
        returns = self.market_data['close'].pct_change()
        self.volatility_series = returns.rolling(window=window, min_periods=1).std()
        self.volatility_series.fillna(0.1, inplace=True)
        
    
    # RESET ENVIRONMENT
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """RESET ENV TO INITIAL STATE FOR NEW EPISODE"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.inventory = self.parent_order_size
        self.cash_spent = 0.0
        self.shares_executed = 0
        self.execution_prices = []
        self.execution_history = []
        observation = self._get_observation()
        info = self._get_info()
        
        logger.debug(f"Environment reset for new episode. INV = {self.inventory}")
        return observation, info
    
    
    def _get_observation(
        self
    ) ->  np.ndarray: 
        """CONSTRUCT 8 DIM OBSERVATION VECTOR
        
            STATE SPACE (8 DIM):
        1. NORM BEST BID (PROXY VWAP)
        2. NORM BEST ASK (PROXY VWAP)
        3. NORM SPREAD
        4. TIME PROGRESS (0 TO 1)
        5. INV REMAINING (0 TO 1)
        6. CURRENT VOLATILITY
        7. EXECUTION PROGRESS (0 TO 1)
        8. NORM AVG EXECUTION PRICE
        """
        step = min(self.current_step, len(self.market_data) - 1)    # ENSURE STEP VALID, SELECT MIN BAR 
        
        # FETCH BAR DATA
        bar = self.market_data.iloc[step]
        current_price = bar['close']                                # ASSIGNS 'CLOSE' COLUMN PRICE
        current_vwap = bar['vwap']                                  # ASSIGNS 'VWAP' COLUMN 
        
        
        # 1,2,3 - SYNTHETIC BEST BID/ASK USING SPREAD FACTOR
        spread = current_price * self.spread_factor
        best_bid = current_vwap - (spread / 2)
        best_ask = current_vwap + (spread / 2)
        
        if best_bid < 0:
            best_bid = current_price * 0.99
        if best_ask <= best_bid:
            best_ask = best_bid * 1.001
        spread = best_ask - best_bid
        
        # 4,5 - TIME & INVENTORY
        time_progress = self.current_step / max(self.time_horizon, 1)   
        inventory_remaining = self.inventory / self.parent_order_size
        
        # 6 - CURRENT VOLATILITY
        current_volatility = self.volatility_series.iloc[step]
        
        # 7 - EXECUTION PROGRESS
        execution_progress = self.shares_executed / self.parent_order_size
        
        # 8 - EXECUTION PRICE
        if self.shares_executed > 0 :
            avg_execution_price = self.cash_spent / self.shares_executed
        else:
            avg_execution_price = current_price
        
        # NORMALISE
        if current_price == 0:
            current_price = 1.0  # Prevent division by zero
        
        best_bid_norm = best_bid / current_price
        best_ask_norm = best_ask / current_price
        spread_norm = spread / current_price
        avg_execution_price_norm = avg_execution_price / current_price
        
        # STATE VECOTOR:
        state = np.array([
            best_bid_norm,
            best_ask_norm,
            spread_norm,
            time_progress,
            inventory_remaining,
            current_volatility,
            execution_progress,
            avg_execution_price_norm
        ], dtype=np.float32)
        
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        # HANDLE NaN/INF SAFETY BEFORE RETURNING STATE ARRAY
        return state
               
        
    # STEP FUNCTION  
    def step(
        self,
        action : int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """ TAKE 1 STEP IN ENV (EXECUTE ACTION) 
        
            ACTION SPACE (3 DIM):
        0. WAIT (NO ACTION)
        1. EXECUTE 10% OF REMAINING INV
        2. EXECUTE 5% OF REMAINING INV  """
        # DETERMINE ORDER SIZE BASED ON ACTION
        if action == 1 and self.inventory > 0:
            order_size = max(1, int(self.inventory * 0.10))  # EXECUTE 10% OF REMAINING INV
            order_size = min(order_size, self.inventory)     # HANDLE CASE WHERE REMAINING INV < ORDER SIZE
        elif action == 2 and self.inventory > 0:
            order_size = max(1, int(self.inventory * 0.05))  # EXECUTE 5% OF REMAINING INV
            order_size = min(order_size, self.inventory)     # HANDLE CASE WHERE REMAINING INV < ORDER SIZE
        else:
            order_size = 0                                   # WAIT ACTION OR NO INV REMAINING
        
        # EXECUTE ORDER
        if order_size > 0:
            step = min(self.current_step, len(self.market_data) - 1)        # ENSURE STEP VALID, SELECT MIN BAR 
            bar = self.market_data.iloc[step]
            
            base_price = bar['vwap']                                        # USE VWAP AS EXECUTION PRICE BASIS
            execution_price = base_price * (1.0 + self.execution_cost_bps)  # APPLY SLIPPAGE/FEES
            
            # RECORD
            self.inventory -= order_size                     # UPDATE REMAINING INV
            self.cash_spent += order_size * execution_price  # UPDATE CASH SPENT
            self.shares_executed += order_size               # UPDATE SHARES EXECUTED
            self.execution_prices.append(execution_price)    # RECORD EXECUTION PRICE
            self.execution_history.append(                   # RECORD EXECUTION HISTORY
                {
                    'step': self.current_step,
                    'quantity': order_size,
                    'price': execution_price,
                    'vwap': base_price
                }
            )
            
        # ADVANCE STEP
        self.current_step += 1
        # REWARD CALCULATION
        reward = self._calculate_reward()
        # CHECK TERMINATION
        terminated = (self.current_step >= self.time_horizon) or (self.inventory <= 0)
        truncated = False
        # GET NEXT OBSERVATION
        next_obs = self._get_observation()
        
        info = self._get_info()
        
        return next_obs, reward, terminated, truncated, info



    # CALCULATE REWARD FUNCTION
    def _calculate_reward(
        self
    ) -> float:
        """ CALCULATE STEP REWARD BASED ON EXECUTION PERFORMANCE 
        
        NEGATIVE SLIPPAGE VS VWAP (MAIN)
        - FOR INVENTORY REMAINING AT DEADLINE
        + FOR ORDER COMPLETION
        """
        if self.shares_executed == 0:
            return -0.01 # PENALTY FOR NO EXECUTIONS
        
        # VWAP BENCHMARK
        step = min(self.current_step, len(self.market_data) - 1)        # ENSURE STEP VALID, SELECT MIN BAR
        vwap_benchmark = self.market_data['vwap'].iloc[:step + 1].mean()
        
        # AVG EXECUTION PRICE
        avg_execution_price = self.cash_spent / self.shares_executed
        
        # SLIPPAGE CALCULATION
        slippage_bps = (avg_execution_price - vwap_benchmark) / vwap_benchmark * 10000.0
        
        # REWARD IS NEGATIVE SLIPPAGE
        reward = -slippage_bps / 100.0  # CONVERT BPS TO DECIMAL
        
        # DEADLINE PENALTY
        if self.current_step >= self.time_horizon - 1 and self.inventory > 0:
            penalty = (self.inventory / self.parent_order_size) * 10.0  # PENALTY FOR REMAINING INV
            reward -= penalty
        
        # COMPLETION BONUS
        if self.inventory == 0:
            reward += 1.0  # BONUS FOR FULL EXECUTION
        
        return reward
        
            
        
        
        
    def _get_info(self) -> Dict:
        """Return info dict with metrics."""
        
        # Calculate performance metrics
        if self.shares_executed > 0:
            avg_exec_price = self.cash_spent / self.shares_executed
            
            # VWAP benchmark
            step = min(self.current_step, len(self.market_data) - 1)
            vwap_benchmark = self.market_data['vwap'].iloc[:step+1].mean()
            
            slippage_bps = ((avg_exec_price - vwap_benchmark) / vwap_benchmark) * 10000
        else:
            avg_exec_price = 0.0
            vwap_benchmark = 0.0
            slippage_bps = 0.0
        
        return {
            'current_step': self.current_step,
            'inventory': self.inventory,
            'shares_executed': self.shares_executed,
            'cash_spent': self.cash_spent,
            'avg_execution_price': avg_exec_price,
            'vwap_benchmark': vwap_benchmark,
            'slippage_bps': slippage_bps,
            'execution_history': self.execution_history.copy(),
            'completion_rate': self.shares_executed / self.parent_order_size
        }
    
        
        
        
        
        

        
        
        