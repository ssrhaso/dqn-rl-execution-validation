"""
TIME WEIGHTED AVERAGE PRICE (TWAP) POLICY
UNIFORMLY OVER TIME HORIZON
"""

import numpy as np
from typing import Tuple

class TWAPPolicy:
    """ EXECUTE EQUAL PORTION OF INV AT REGULAR , TIMED INTERVALS """
    
    def __init__(
        self,
        parent_order_size: int,
        time_horizon: int,
    ):
        self.parent_order_size = parent_order_size
        self.time_horizon = time_horizon
        self.shares_per_step = parent_order_size / time_horizon
        self.step_count = 0
        
        
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, None]:
        """ RETURN ACTION BASED ON UNIFORM TIME SCHEDULE """
        self.step_count += 1
        
        # EXTRACT INV
        inventory_remaining_pct = observation[4]    # 5TH ELEMENT IS INVENTORY % REMAINING
        time_progress = observation[3]              # 4TH ELEMENT IS TIME PROGRESS 
        
        # SMART EXECUTION LOGIC
        if inventory_remaining_pct < 0.05:
            return 0, None # WAIT IF LESS THAN 5% INV REMAINS
        
        if time_progress > 0.8:
            return 1, None # EXECUTE 10% IF 80% TIME ELAPSED
        return 2, None     # OTHERWISE EXECUTE 20% OF PARENT ORDER
    

    def reset(
        self
    ) -> None: 
        """ RESET FOR NEW EPISODE """
        self.step_count = 0
        
        
        
        