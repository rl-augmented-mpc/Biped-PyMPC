import torch

class GaitGenerator:
    def __init__(self, 
                 batch_size: int, 
                 mpc_horizon: int, 
                 dt: float,
                 dt_mpc: torch.Tensor, 
                 dsp_durations: torch.Tensor, 
                 ssp_durations: torch.Tensor, 
                 device: torch.device = torch.device("cpu")):
        """
        Gait generator of bipedal robot. 
        
        Args:
            batch_size (int): Number of batch elements for parallel computation
            mpc_horizon (int): Number of future timesteps to plan
            dt (float): Control time step in sec
            dt_mpc (torch.Tensor): Tensor of shape (num_envs,) representing the MPC discretization timestep in sec
            dsp_durations (torch.Tensor([batch_size, 2])): Double support phase durations [left foot, right foot] in int
            ssp_durations (torch.Tensor([batch_size, 2])): Single support phase durations [left foot, right foot] in int
            device (torch.device): Device to run the calculations on (default: CPU)
            
        Gait looks like the following:
        =======================================================================
        Left foot: stance | double stance | swing | double stance | stance |...
        Right foot: swing | double stance | stance | double stance | swing |...
        =======================================================================
        dsp_durations (shape: [batch_size, 2]) controls the duration of double stance phase for each foot (left-right order).
        ssp_durations (shape: [batch_size, 2]) controls the duration of single stance phase for each foot (left-right order).
        
        Our gait is phase-based, meaning every control step, phase is incremented by:
        \phi_t+1 = \phi_t + \Delta t / gait_cycle_length_s
        where phi_t is phase, \Delta t is time step in sec, 
        and gait_cycle_length_s is the total gait cycle length in sec.
        
        sub-phases are calculated as follows:
        - contact_sub_phase: 0 for swing phase, 0 to 1 for stance phase
        - swing_sub_phase: 0 for stance phase, 0 to 1 for swing phase
        """
        
        self.batch_size = batch_size
        self.mpc_horizon = mpc_horizon
        self.dt = dt
        self.dt_mpc = dt_mpc
        self.device = device

        self.dsp_durations = dsp_durations.to(torch.int32).to(device)
        self.ssp_durations = ssp_durations.to(torch.int32).to(device)

        # Gait cycle length which is the sum of all durations
        self.gait_cycle_length = torch.sum(self.dsp_durations+self.ssp_durations, dim=1) # (batch_size,)
        self.gait_phase = torch.zeros(self.batch_size, device=self.device)

        # stance and swing durations during one gait cycle
        self.stance_durations = torch.stack([
            self.ssp_durations[:,1] + torch.sum(self.dsp_durations, dim=1),  # left foot stance durations
            self.ssp_durations[:,0] + torch.sum(self.dsp_durations, dim=1)   # right foot stance durations
        ], dim=1)
        
        self.swing_durations = torch.stack([
            self.ssp_durations[:, 0],  # left foot swing durations
            self.ssp_durations[:, 1]   # right foot swing durations
        ], dim=1)
        
        # Convert to the unit of each foot duration from control time step(::int) to phase(::float)
        self.ssp_durations_phase = self.ssp_durations / self.gait_cycle_length.unsqueeze(1) 
        self.dsp_durations_phase = self.dsp_durations / self.gait_cycle_length.unsqueeze(1) 
        
        # Contact state lookup table for MPC constraints
        self.mpc_table = torch.ones((batch_size, mpc_horizon, 2), dtype=torch.int32, device=self.device)

        # each durations in sec
        self.swing_durations_sec = self.swing_durations * self.dt_mpc[:, None]
        self.stance_durations_sec = self.stance_durations * self.dt_mpc[:, None]
        self.gait_durations_sec = self.gait_cycle_length * self.dt_mpc

    def reset(self, env_ids: torch.Tensor = None):
        """Reset the gait phase"""
        self.gait_phase[env_ids] = 0.0

    def update_phase(self):
        """
        Update the gait phase based on stepping frequency
        Args:
            stepping_frequency: How fast the gait should progress (1.0 = normal speed)
        """
        delta_t = 1
        self.gait_phase += self.dt / self.gait_durations_sec
        self.gait_phase -= (self.gait_phase > 1).float() * 1.0 # If gait phase >= 1, subtract 1

    def update_sampling_time(self, dt_mpc: torch.Tensor):
        """
        update MPC sampling time.
        """
        self.dt_mpc = dt_mpc
        self.swing_durations_sec = self.swing_durations * self.dt_mpc[:, None]
        self.stance_durations_sec = self.stance_durations * self.dt_mpc[:, None]
        self.gait_durations_sec = self.gait_cycle_length * self.dt_mpc

    def get_contact_sub_phase(self) -> torch.Tensor:
        """Calculate the contact sub-phase for each foot across all batch elements
        Returns:
            A tensor containing the contact sub-phase for the left and right feet.
            If a foot is in swing phase, its contact sub-phase will be 0.
        """
        self.contact_sub_phase = -torch.ones((self.batch_size, 2), device=self.device)
        
        # Left foot
        # Phase 1: Halfway through, when the robot finishes left SSP and the first DSP
        phase_threshold_1 = self.ssp_durations_phase[:, 0] + self.dsp_durations_phase[:, 0]
        # Phase 2: When the robot finishes right SSP and is ready to start the second DSP
        phase_threshold_2 = phase_threshold_1 + self.ssp_durations_phase[:, 1]
        
        # Create masks for different phases
        # Left foot in stance phase + first DSP
        mask_1 = self.gait_phase < phase_threshold_1
        # Final DSP, transitioning to next cycle
        mask_2 = self.gait_phase >= phase_threshold_2
        
        # Apply calculations using masks
        self.contact_sub_phase[mask_1, 0] = self.gait_phase[mask_1] / phase_threshold_1[mask_1]
        # mask_2 corresponds to swing phase, where contact_sub_phase remains 0
        # self.dsp_durations_phase[0] or self.dsp_durations_phase[1] is ok. dsp should be symmetric for both feet
        self.contact_sub_phase[mask_2, 0] = (self.gait_phase[mask_2] - phase_threshold_2[mask_2]) / self.dsp_durations_phase[mask_2,0]
        
        # Right foot
        right_threshold = self.ssp_durations_phase[:, 1]
        mask_right_swing = self.gait_phase < right_threshold
        mask_right_stance = ~mask_right_swing
        
        # Right foot stance calculation
        # self.dsp_durations_phase[0] or self.dsp_durations_phase[1] is ok. dsp should be symmetric for both feet
        self.contact_sub_phase[mask_right_stance, 1] = (
            (self.gait_phase[mask_right_stance] - right_threshold[mask_right_stance]) / 
            (self.dsp_durations_phase[mask_right_stance, 0] + self.ssp_durations_phase[mask_right_stance, 1] + self.dsp_durations_phase[mask_right_stance, 1])
        )
        
        return self.contact_sub_phase
    
    def get_swing_sub_phase(self) -> torch.Tensor:
        """
        Calculate the swing sub-phase for each foot.
        
        Returns:
            A tensor containing the swing sub-phase for the left and right feet.
            If a foot is in stance phase, its swing sub-phase will be 0.
        """
        swing_sub_phase = -torch.ones((self.batch_size, 2), device=self.device)
        
        # Left foot
        left_swing_start = self.ssp_durations_phase[:, 1] + self.dsp_durations_phase[:, 0]
        left_swing_end = left_swing_start + self.ssp_durations_phase[:, 0]
        mask_left_swing = torch.logical_and(
            self.gait_phase >= left_swing_start,
            self.gait_phase < left_swing_end
        )
        
        swing_sub_phase[mask_left_swing, 0] = (
            (self.gait_phase[mask_left_swing] - left_swing_start[mask_left_swing]) / 
            self.ssp_durations_phase[mask_left_swing, 0]
        )
        
        # Right foot
        mask_right_swing = self.gait_phase < self.ssp_durations_phase[:, 1]
        swing_sub_phase[mask_right_swing, 1] = (
            self.gait_phase[mask_right_swing] / 
            self.ssp_durations_phase[mask_right_swing, 1]
        )
        
        return swing_sub_phase
    
    """
    properties.
    """
    
    @property
    def contact_bool(self) -> torch.Tensor:
        """
        Get the contact boolean for each foot across all batch elements
        Returns:
            A tensor containing the contact boolean for the left and right feet.
            If a foot is in stance phase, its contact boolean will be True.
        """
        contact_bool = torch.zeros((self.batch_size, 2), dtype=torch.bool, device=self.device)
        contact_sub_phase = self.get_contact_sub_phase()
        
        # Left foot
        contact_bool[:, 0] = contact_sub_phase[:, 0] != -1
        
        # Right foot
        contact_bool[:, 1] = contact_sub_phase[:, 1] != -1
        
        return contact_bool
    
    @property
    def swing_bool(self) -> torch.Tensor:
        """
        Get the swing boolean for each foot across all batch elements
        Returns:
            A tensor containing the swing boolean for the left and right feet.
            If a foot is in swing phase, its swing boolean will be True.
        """
        swing_bool = torch.zeros((self.batch_size, 2), dtype=torch.bool, device=self.device)
        swing_sub_phase = self.get_swing_sub_phase()
        
        # Left foot
        swing_bool[:, 0] = swing_sub_phase[:, 0] != -1
        
        # Right foot
        swing_bool[:, 1] = swing_sub_phase[:, 1] != -1
        
        return swing_bool
    
    @property
    def mpc_gait(self) -> torch.Tensor:
        """
        Generate the gait pattern for the MPC horizon

        Returns:
            mpc_table: A tensor containing the contact sequence for each foot during the MPC horizon (batch_size, mpc_horizon, 2)
        """
        gait_time_step_from_phase = (self.gait_phase * self.gait_cycle_length).int() # (batch_size,)
        
        # Create time step bin during mpc horizon window
        time_steps = torch.arange(self.mpc_horizon, 
                                  device=self.device).unsqueeze(0).repeat(self.batch_size, 1) # (batch_size, mpc_horizon)
        gait_steps = (gait_time_step_from_phase.unsqueeze(1) + time_steps) % self.gait_cycle_length.unsqueeze(1) # (batch_size, mpc_horizon)
        
        # Create masks for different phases
        phase1_mask = gait_steps < self.ssp_durations[:, 1].unsqueeze(1) # Left stance, right swing
        phase2_mask = torch.logical_and(
            gait_steps >= self.ssp_durations[:, 1].unsqueeze(1),
            gait_steps < self.ssp_durations[:, 1].unsqueeze(1) + self.dsp_durations[:, 0].unsqueeze(1)
        ) # Double support
        phase3_mask = torch.logical_and(
            gait_steps >= self.ssp_durations[:, 1].unsqueeze(1) + self.dsp_durations[:, 0].unsqueeze(1),
            gait_steps < self.ssp_durations[:, 1].unsqueeze(1) + self.dsp_durations[:, 0].unsqueeze(1) + self.ssp_durations[:, 0].unsqueeze(1)
        ) # Left swing, right stance
        
        # Initialize contact patterns
        self.mpc_table.zero_()
        
        # Set contact patterns based on phases
        self.mpc_table[:, :, 0][phase1_mask] = 1  # Left stance, right swing
        self.mpc_table[:, :, 0][phase2_mask]= 1  # Double support
        self.mpc_table[:, :, 1][phase2_mask] = 1 # Double support
        self.mpc_table[:, :, 1][phase3_mask] = 1  # Left swing, right stance
        self.mpc_table[:, :, 0][~(phase1_mask | phase2_mask | phase3_mask)] = 1  # Final double support
        self.mpc_table[:, :, 1][~(phase1_mask | phase2_mask | phase3_mask)] = 1 # Final double support

        return self.mpc_table


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    device = torch.device("cuda")
    batch_size = 4096
    mpc_horizon = 10
    dt = 0.001
    dt_mpc = 40 * dt * torch.ones(batch_size, device=device)
    dsp_durations = torch.tensor([0.05, 0.05], device=device).unsqueeze(0).repeat(batch_size, 1)
    ssp_durations = torch.tensor([0.2, 0.2], device=device).unsqueeze(0).repeat(batch_size, 1)
    gait = GaitGenerator(
        batch_size=batch_size,
        mpc_horizon=10,
        dt = dt,
        dt_mpc=dt_mpc,
        dsp_durations=dsp_durations,
        ssp_durations=ssp_durations, 
        device=device
    )
    
    batch_idx = 0
    time_e = 0.4+0.1
    time_steps = torch.linspace(0, time_e, int(time_e/dt))
    
    gait_phase = []
    left_contact_phase = []
    right_contact_phase = []
    left_swing_phase = []
    right_swing_phase = []
    
    for t in time_steps:
        gait.update_phase(1.0)
        contact_sub_phase = gait.get_contact_sub_phase()
        swing_sub_phase = gait.get_swing_sub_phase()
        
        gait_phase.append(gait.gait_phase[batch_idx].item())
        left_contact_phase.append(contact_sub_phase[batch_idx, 0].item())
        right_contact_phase.append(contact_sub_phase[batch_idx, 1].item())
        left_swing_phase.append(swing_sub_phase[batch_idx, 0].item())
        right_swing_phase.append(swing_sub_phase[batch_idx, 1].item())
    
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 2, 1)
    plt.plot(time_steps, gait_phase)
    plt.title("Gait Phase")
    plt.xlabel("time (s)")
    plt.ylabel(r"$\phi$")
    
    plt.subplot(3, 2, 3)
    plt.plot(time_steps, left_contact_phase)
    plt.title("Left Contact Phase")
    plt.xlabel("time (s)")
    plt.ylabel(r"$\phi$")
    
    plt.subplot(3, 2, 4)
    plt.plot(time_steps, right_contact_phase)
    plt.title("Right Contact Phase")
    plt.xlabel("time (s)")
    plt.ylabel(r"$\phi$")
    
    plt.subplot(3, 2, 5)
    plt.plot(time_steps, left_swing_phase)
    plt.title("Left Swing Phase")
    plt.xlabel("time (s)")
    plt.ylabel(r"$\phi$")
    
    plt.subplot(3, 2, 6)
    plt.plot(time_steps, right_swing_phase)
    plt.title("Right Swing Phase")
    plt.xlabel("time (s)")
    plt.ylabel(r"$\phi$")
    
    plt.tight_layout()
    plt.show()