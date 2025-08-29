import torch
from biped_pympc.biped_controller import BipedController, MPCConf, ControllerConf

class MPCController:
    """
    MPC controller high-level wrapper for biped_controller.
    """
    def __init__(self, cfg: ControllerConf, mpc_cfg:MPCConf, num_envs:int, device:torch.device, gait_id:int=1):
        self.num_envs = num_envs
        self.device = device
        num_legs = 2
        self.biped_controller = BipedController(cfg, mpc_cfg, num_envs, num_legs, device, gait_id)
    
    """ 
    operations.
    """
    def set_command(self, twist:torch.Tensor, height:torch.Tensor)->None:
        self.biped_controller.set_desired_state(twist, height)
        
    def update_state(self, state:torch.Tensor)->None:
        actuated_leg_dof = 2 * self.biped_controller.leg_controller.biped.num_dof # TODO: looks bit ugly
        position = state[:, :3]
        orientation = state[:, 3:7]
        lin_velocity = state[:, 7:10]
        ang_velocity = state[:, 10:13]
        joint_pos = state[:, 13:13 + actuated_leg_dof]
        joint_vel = state[:, 13 + actuated_leg_dof:13 + 2*actuated_leg_dof]
        joint_toque = state[:, 13 + 2*actuated_leg_dof:13 + 3*actuated_leg_dof]
        
        self.biped_controller.set_leg_data(joint_pos, joint_vel, joint_toque)
        self.biped_controller.get_state_estimate(position, orientation, lin_velocity, ang_velocity)
        
    def run_mpc(self)->None:
        self.biped_controller.run_mpc()
    
    def run_lowlevel(self)->None:
        self.biped_controller.run_lowlevel()
    
    def get_action(self)->torch.Tensor:
        return self.biped_controller.command_joint_torque.clone()
    
    def reset(self, env_ids: torch.Tensor)->None:
        self.biped_controller.reset(env_ids)

    """
    DRL interface.
    """
    def update_mpc_sampling_time(self, dt_mpc:torch.Tensor)->None:
        """
        update MPC discretization timestep. 

        Args:
            dt_mpc (torch.Tensor): Tensor of shape (num_envs,) representing the new sampling time for each environment.
        """
        self.biped_controller.gait_generator.update_sampling_time(dt_mpc)
        self.biped_controller.dt_mpc = dt_mpc
    
    def set_swing_parameters(self, foot_height: torch.Tensor, cp1:torch.Tensor, cp2:torch.Tensor)->None:
        self.biped_controller.swing_leg_controller.set_foot_height(foot_height)
        self.biped_controller.swing_leg_controller.set_control_points(cp1, cp2)

    def set_srbd_accel(self, residual_lin_accel:torch.Tensor, residual_ang_accel:torch.Tensor)->None:
        self.biped_controller.mpc_controller.residual_lin_accel = residual_lin_accel.clone()
        self.biped_controller.mpc_controller.residual_ang_accel = residual_ang_accel.clone()
    
    def set_srbd_residual(self, A_residual:torch.Tensor, B_residual:torch.Tensor)->None:
        raise NotImplementedError
    
    """ 
    properties.
    """
    @property
    def ground_reaction_wrench(self)->torch.Tensor:
        """
        ground reaction wrench in base frame (num_envs, num_legs, 6)
        """
        return self.biped_controller.leg_controller.command.feedfowardforce.clone()
    
    @property
    def centroidal_accel(self)->torch.Tensor:
        """
        centroidal acceleration in base frame (num_envs, 6)
        """
        wrench = self.ground_reaction_wrench # (num_envs, num_legs, 6)
        accel = torch.zeros((self.num_envs, 6), device=self.device)
        accel[:, :3] = torch.sum(wrench[:, :, :3], dim=1) / self.biped_controller.mpc_controller.mass
        accel[:, 3:] = (self.biped_controller.mpc_controller.I_world_inv[:, 0, :, :] @ torch.sum(wrench[:, :, 3:], dim=1)[:, :, None]).squeeze(-1)
        return accel
    
    @property
    def contact_state(self)->torch.Tensor:
        contact_phase = self.biped_controller.contact_phase
        contact_state = torch.where(contact_phase == -1, torch.zeros_like(contact_phase), torch.ones_like(contact_phase))
        return contact_state
    
    @property
    def contact_phase(self)->torch.Tensor:
        contact_phase = self.biped_controller.contact_phase
        contact_state = torch.where(contact_phase == -1, torch.zeros_like(contact_phase), torch.ones_like(contact_phase))
        return contact_state * contact_phase
    
    @property
    def swing_state(self)->torch.Tensor:
        swing_phase = self.biped_controller.swing_phase
        swing_state = torch.where(swing_phase == -1, torch.zeros_like(swing_phase), torch.ones_like(swing_phase))
        return swing_state
    
    @property
    def swing_phase(self)->torch.Tensor:
        swing_phase = self.biped_controller.swing_phase
        swing_state = torch.where(swing_phase == -1, torch.zeros_like(swing_phase), torch.ones_like(swing_phase))
        return swing_state * swing_phase
    
    @property
    def foot_placement(self)->torch.Tensor:
        """ 
        planned footholds (num_envs, num_legs, 3)
        """
        return self.biped_controller.swing_leg_controller.foot_placement.clone()

    @property
    def foot_placement_b(self)->torch.Tensor:
        """
        foot placement in base frame (num_envs, num_legs, 3)
        """
        return self.biped_controller.swing_leg_controller.foot_placement_b.clone()

    
    @property
    def ref_foot_pos_b(self)->torch.Tensor:
        """ 
        reference foot position in body frame (num_envs, num_legs, 3)
        """
        return self.biped_controller.leg_controller.command.pDes.clone()
    
    @property
    def ref_foot_vel_b(self)->torch.Tensor:
        """ 
        reference foot velocity in body frame (num_envs, num_legs, 3)
        """
        return self.biped_controller.leg_controller.command.vDes.clone()
    
    @property
    def foot_pos_b(self)->torch.Tensor:
        return self.biped_controller.leg_controller.data.p.clone()
    
    @property
    def foot_vel_b(self)->torch.Tensor:
        return self.biped_controller.leg_controller.data.v.clone()
    
    @property
    def mpc_cost(self)->torch.Tensor:
        return self.biped_controller.mpc_cost.clone()
    
    @property
    def position_trajectory(self)->torch.Tensor:
        return self.biped_controller.mpc_controller.x_ref[:, :, :3].clone()
    
    @property
    def velocity_trajectory(self)->torch.Tensor:
        return self.biped_controller.mpc_controller.x_ref_dot.clone()
    
    @property
    def swing_foot_trajectory(self)->torch.Tensor:
        num_samples = 10 
        phase_batch = torch.linspace(0, 1, num_samples, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        left_foot_trajectory = torch.zeros((self.num_envs, num_samples, 3), device=self.device)
        right_foot_trajectory = torch.zeros((self.num_envs, num_samples, 3), device=self.device)
        swing_leg_trajectory = torch.zeros((self.num_envs, num_samples, 3), device=self.device)

        rotation_matrix = self.biped_controller.state_estimator.data.rotation_body  # (num_envs, 3, 3)
        robot_position = self.biped_controller.state_estimator.data.root_position  # (num_envs, 3)

        for i in range(num_samples):
            phase = phase_batch[:, i]
            
            # left leg
            self.biped_controller.swing_leg_controller.swing_leg_trajectory[0].compute_swing_trajectory(
                phase,
                self.biped_controller.swing_leg_controller.swing_duration[:, 0]
            )

            if self.biped_controller.swing_leg_controller.reference_frame == 'world':
                traj = self.biped_controller.swing_leg_controller.swing_leg_trajectory[0].get_position()
                left_foot_trajectory[:, i, :] = (rotation_matrix.transpose(1, 2) @ (traj - robot_position).unsqueeze(2)).squeeze(2)
            elif self.biped_controller.swing_leg_controller.reference_frame == 'base':
                left_foot_trajectory[:, i, :] = self.biped_controller.swing_leg_controller.swing_leg_trajectory[0].get_position()

            # right leg
            self.biped_controller.swing_leg_controller.swing_leg_trajectory[1].compute_swing_trajectory(
                phase,
                self.biped_controller.swing_leg_controller.swing_duration[:, 1]
            )
            if self.biped_controller.swing_leg_controller.reference_frame == 'world':
                traj = self.biped_controller.swing_leg_controller.swing_leg_trajectory[1].get_position()
                right_foot_trajectory[:, i, :] = (rotation_matrix.transpose(1, 2) @ (traj - robot_position).unsqueeze(2)).squeeze(2)
            elif self.biped_controller.swing_leg_controller.reference_frame == 'base':
                right_foot_trajectory[:, i, :] = self.biped_controller.swing_leg_controller.swing_leg_trajectory[1].get_position()

        swing_leg_trajectory = \
            left_foot_trajectory * (1 - self.biped_controller.leg_controller.data.contact_bool[:, 0, None, None]) + \
            right_foot_trajectory * (1 - self.biped_controller.leg_controller.data.contact_bool[:, 1, None, None])

        return swing_leg_trajectory