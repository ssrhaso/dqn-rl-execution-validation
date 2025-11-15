"""
VOLUME WEIGHTED AVERAGE PRICE (TWAP) POLICY
UNIFORMLY OVER TIME HORIZON
"""

import numpy as np
from typing import Tuple

class VWAPPolicy:
    """ EXECUTE PROPORTIONAL TO EXPECTED VOLUME PROFILE """
    
    def __init__(
        self,
        parent_order_size: int,
    ):
        self.parent_order_size = parent_order_size
        self.cumulative_volume = 0
        self.estimated_total_volume = 0
        
        
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, None]:
        """ RETURN ACTION BASED ON VOLUME WEIGHTED SCHEDULE 
        USE VOLATILITY AS PROXY FOR VOLUME 
        (HIGH VOL = HIGH ACTIVITY = SLOWER EXECUTION) 
        (LOW VOL = LOW ACTIVITY = FASTER EXECUTION)
        SINCE NO DIRECT ACCESS TO VOLUME DATA IN ALPACA FREE TIER
        """
        
        # EXTRACT INV
        inventory_remaining_pct = observation[4]    # 5TH ELEMENT IS INVENTORY % REMAINING
        time_progress = observation[3]              # 4TH ELEMENT IS TIME PROGRESS 
        volatility = observation[5]                 # 6TH ELEMENT IS VOLATILITY
        
        # SMART EXECUTION LOGIC
        if inventory_remaining_pct < 0.05:
            return 0, None # WAIT IF LESS THAN 5% INV REMAINS
        
        if time_progress > 0.85:
            return 1, None # EXECUTE 10% IF 85% TIME ELAPSED
        
    
        # VOLUME WEIGHTED EXECUTION
        if volatility > 0.6:
            return 0, None # HIGH VOL = WAIT
        elif volatility > 0.3:
            return 2, None # MEDIUM VOL = EXECUTE 5%
        else:
            return 1, None # LOW VOL = EXECUTE 10%
    

    def reset(
        self
    ) -> None: 
        """ RESET FOR NEW EPISODE """
        self.cumulative_volume = 0
        
        
        
        