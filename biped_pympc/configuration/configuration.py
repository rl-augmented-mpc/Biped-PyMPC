from dataclasses import dataclass, field
from typing import Literal
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ControllerConf:
    """
    Configuration parameters for the biped controller.
    ssp_durations: float = duration of single support phase in seconds.
    dsp_durations: float = duration of double support phase in seconds.
    swing_height: float = height of the swing leg in meters.
    torque_limit: list[float] = list of torque limits for each joint in Nm.
    """
    
    ssp_durations: float = 0.2  # seconds
    dsp_durations: float = 0.0 # seconds
    swing_height: float = 0.1  # meters
    swing_reference_frame: Literal["world", "base"] = "base"

@dataclass
class MPCConf:
    """ 
    MPC configuration parameters.
    dt: float = control time step in seconds.
    iteration_between_mpc: int = number of iterations between MPC updates.
    horizon_length: int = length of the prediction horizon in steps.
    decimation: int = number of control steps to skip between MPC updates.
    Q: torch.Tensor = state tracking cost matrix.
    R: torch.Tensor = control tracking cost matrix.
    """
    
    dt: float = 0.001
    dt_mpc: float = 40 * 0.001
    
    horizon_length: int = 10
    decimation: int = 10 # how many control steps to skip between mpc updates
    
    Q: torch.Tensor = torch.tensor(
        # [200, 500, 500, 500, 500, 500, 1, 1, 5, 1, 1, 5, 1], 
        [150, 150, 250, 100, 100, 250, 1, 1, 5, 10, 10, 1, 1], 
        # [150, 300, 250, 100, 100, 250, 1, 1, 5, 10, 10, 1, 1], # T1
        device=DEVICE, dtype=torch.float32)
    R: torch.Tensor = torch.tensor(
        # [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2], 
        [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4], 
        # [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-5, 1e-4, 1e-4, 1e-5, 1e-4], # T1
        device=DEVICE, dtype=torch.float32)
    
    print_solve_time: bool = False
    solver: Literal["osqp", "qpth", "casadi-ipm", "qpswift"] = "osqp"

    robot: Literal["HECTOR", "T1"] = "HECTOR"