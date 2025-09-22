from dataclasses import dataclass, field
from typing import Literal
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ControllerConf:
    """
    Configuration parameters for the biped controller.

    ssp_durations: int = duration of single support phase in mpc steps.
    dsp_durations: int = duration of double support phase in mpc steps.
    swing_height: float = height of the swing leg in meters.
    swing_reference_frame: Literal["world", "base"] = reference frame for swing leg trajectory.
    """
    
    ssp_durations: int = 5
    dsp_durations: int = 0
    swing_height: float = 0.1  # meters
    swing_reference_frame: Literal["world", "base"] = "base"

@dataclass
class MPCConf:
    """ 
    MPC configuration parameters.
    dt: float = control time step in seconds.
    dt_mpc: float = MPC time step in seconds.
    horizon_length: int = length of the prediction horizon in steps.
    decimation: int = number of control steps to skip between MPC updates.
    Q: torch.Tensor = state tracking cost matrix.
    R: torch.Tensor = control tracking cost matrix.
    print_solve_time: bool = whether to print the solver time.
    solver: Literal["osqp", "qpth", "casadi", "cusadi"] = choice of QP solver.
    robot: Literal["HECTOR", "T1"] = robot model to use.
    """
    
    dt: float = 0.001
    dt_mpc: float = 0.025
    
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
    
    print_solve_time: bool = True
    solver: Literal["osqp", "qpth", "casadi", "cusadi"] = "cusadi"

    robot: Literal["HECTOR", "T1"] = "HECTOR"

    def __post_init__(self):
        print('[INFO] MPC Configuration:')
        print('+--------------------------------+')
        print(f'  dt: {self.dt}')
        print(f'  dt_mpc: {self.dt_mpc}')
        print(f'  horizon_length: {self.horizon_length}')
        print(f'  decimation: {self.decimation}')
        print(f'  Q: {self.Q}')
        print(f'  R: {self.R}')
        print(f'  solver: {self.solver}')
        print(f'  robot: {self.robot}')
        print('+--------------------------------+')