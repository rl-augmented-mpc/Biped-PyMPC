from time import time
from typing import Union
from dataclasses import dataclass
import torch

from biped_pympc.configuration.configuration import ControllerConf, MPCConf
from biped_pympc.core.data.robot_data import DesiredStateData
from biped_pympc.core.gait.gait_generator import GaitGenerator
from biped_pympc.controllers.state_estimator import StateEstimator
from biped_pympc.controllers.swing_leg_controller import SwingLegController
from biped_pympc.controllers.leg_controller import LegController

class BipedController:
    def __init__(self, cfg: ControllerConf, mpc_cfg:MPCConf, num_envs:int, num_legs:int, device:Union[torch.device, str], gait_id:int=1):
        self.num_envs = num_envs
        self.num_legs = num_legs
        self.device = device
        
        self.dt = mpc_cfg.dt
        self.dt_mpc = mpc_cfg.dt_mpc * torch.ones(num_envs, device=device)
        self.horizon_length = mpc_cfg.horizon_length
        self.mpc_decimation = mpc_cfg.decimation
        self.print_solve_time = mpc_cfg.print_solve_time

        # # === Standing gait ======
        if gait_id == 1:
            dsp_durations = torch.tensor([5, 5], device=device).unsqueeze(0).repeat(num_envs, 1)
            ssp_durations = torch.tensor([0, 0], device=device).unsqueeze(0).repeat(num_envs, 1)
        elif gait_id == 2:
            # == Walking gait ======
            dsp_durations = torch.tensor([cfg.dsp_durations, cfg.dsp_durations], device=device).unsqueeze(0).repeat(num_envs, 1)
            ssp_durations = torch.tensor([cfg.ssp_durations, cfg.ssp_durations], device=device).unsqueeze(0).repeat(num_envs, 1)
        else:
            raise ValueError(f"Invalid gait_id: {gait_id}. Must be 1 (standing) or 2 (walking).")
        
        if mpc_cfg.solver == "qpth":
            from biped_pympc.convex_mpc.mpc_controller_qpth import MPCControllerQPTh as MPCController
        elif mpc_cfg.solver == "osqp":
            from biped_pympc.convex_mpc.mpc_controller_osqp import MPCControllerOSQP as MPCController
        elif mpc_cfg.solver == "cusadi":
            from biped_pympc.convex_mpc.mpc_controller_cusadi import MPCControllerCusadi as MPCController
        elif mpc_cfg.solver == "casadi":
            from biped_pympc.convex_mpc.mpc_controller_casadi import MPCControllerCasadi as MPCController
        else:
            raise ValueError(f"Invalid solver: {mpc_cfg.solver}. Must be 'qpth', 'osqp', 'casadi', or 'cusadi'.")
        
        self.gait_generator = GaitGenerator(
            batch_size=num_envs,
            mpc_horizon=self.horizon_length,
            dt = self.dt,
            dt_mpc=self.dt_mpc,
            device=device, 
            dsp_durations=dsp_durations, 
            ssp_durations=ssp_durations)
        
        self.desired_state = DesiredStateData(batch_size=num_envs, device=device)
        self.state_estimator = StateEstimator(num_legs=num_legs, batch_size=num_envs, device=device)
        self.leg_controller = LegController(num_envs, num_legs, device, mpc_cfg.robot)
        self.mpc_controller = MPCController(num_envs, device, num_legs, mpc_cfg)
        self.swing_leg_controller = SwingLegController(
            dt=self.dt, 
            batch_size=num_envs, 
            num_legs=num_legs, 
            device=device, 
            swing_duration=self.gait_generator.swing_durations_sec,
            swing_height=cfg.swing_height, 
            reference_frame=cfg.swing_reference_frame,
            robot=mpc_cfg.robot,
            )
        
        ## torque limit
        self.torque_limit = torch.tensor(self.leg_controller.biped.pd_conf.torque_limit, device=device)
        
        # buffer
        self.swing_phase = torch.zeros(num_envs, num_legs, device=device)
        self.contact_phase = torch.zeros(num_envs, num_legs, device=device)
        self.mpc_cost = torch.zeros(num_envs, device=device)
    
    """ 
    reset.
    """
    def reset(self, env_ids:torch.Tensor)->None:
        self.gait_generator.reset(env_ids)
        self.mpc_controller.reset(env_ids)
        self.state_estimator.reset(env_ids)
        self.leg_controller.reset(env_ids)
        self.swing_leg_controller.reset(env_ids)
    
    """
    state estimator.
    """
    def set_desired_state(self, twist:torch.Tensor, height:torch.Tensor):
        self.desired_state.desired_velocity_b[:, :2] = twist[:, :2]
        self.desired_state.desired_angular_velocity_b[:, 2] = twist[:, 2]
        self.desired_state.desired_height = height
    
    def set_leg_data(self, q:torch.Tensor, qd:torch.Tensor, torque:torch.Tensor):
        self.swing_phase = self.gait_generator.get_swing_sub_phase()
        self.contact_phase = self.gait_generator.get_contact_sub_phase()
        self.leg_controller.update_gait_data(self.contact_phase, self.swing_phase)
        self.leg_controller.update_data(q, qd, torque)
    
    def get_state_estimate(self, position:torch.Tensor, orientation:torch.Tensor, lin_velocity:torch.Tensor, ang_velocity:torch.Tensor):
        self.state_estimator.set_body_state(position, orientation, lin_velocity, ang_velocity)
        foot_position_b = torch.cat(
            [
                self.leg_controller.data.p[:, 0, :], 
                self.leg_controller.data.p[:, 1, :]
             ], 
            dim=1) # foot position in body frame
        self.state_estimator.update_foot_position(foot_position_b) # foot position in world frame
    
    """
    control step.
    """
    def run_mpc(self)->None:
        t0 = time()
        self._set_mpc_state()
        self._run_stance_leg_controller()
        t1 = time()
        if self.print_solve_time:
            print("MPC solve time took: ", 1000*(t1 - t0), " ms")
    
    def run_lowlevel(self)->None:
        t0 = time()
        self._run_swing_leg_controller()
        self.leg_controller.update_command()
        
        # update phase and counter
        self.gait_generator.update_phase()

        t1 = time()
        if self.print_solve_time:
            print("low level control took: ", 1000*(t1 - t0), " ms")
    
    def _set_mpc_state(self):
        mpc_gait_table = self.gait_generator.mpc_gait
        self.mpc_controller.set_contact_table(mpc_gait_table)
        self.mpc_controller.set_state_estimate_data(self.state_estimator.data)
        self.mpc_controller.set_desired_state_data(self.desired_state)
        self.mpc_controller.set_leg_controller_data(self.leg_controller.data)
        self.mpc_controller.set_mpc_sampling_time(self.dt_mpc)
    
    def _run_stance_leg_controller(self)->None:
        foot_wrench, mpc_cost = self.mpc_controller.run()
        self.leg_controller.command.feedfowardforce = foot_wrench.reshape(-1, self.num_legs, 6) # avoid direct substitution?
        self.mpc_cost = mpc_cost
    
    def _run_swing_leg_controller(self):
        # update all the states
        self.swing_leg_controller.set_state_estimator(self.state_estimator.data)
        self.swing_leg_controller.set_desired_state(self.desired_state)
        self.swing_leg_controller.set_leg_controller_data(self.leg_controller.data)

        # get gait phase
        swing_phase = self.gait_generator.get_swing_sub_phase()
        contact_phase = self.gait_generator.get_contact_sub_phase()

        # update swing leg controller
        self.swing_leg_controller.update_contact_phase(contact_phase)
        self.swing_leg_controller.update_swing_phase(swing_phase)
        self.swing_leg_controller.update_swing_duration(self.gait_generator.swing_durations_sec)
        self.swing_leg_controller.update_swing_time()
        self.swing_leg_controller.compute_foot_placement()
        p_foot_des, v_foot_des = self.swing_leg_controller.compute_foot_desired_position()
        
        # send command to leg controller
        self.leg_controller.command.pDes = p_foot_des
        self.leg_controller.command.vDes = v_foot_des
    
    @property
    def command_joint_torque(self)->torch.Tensor:
        torque_ff = self.leg_controller.command.tau
        torque_fb = \
            self.leg_controller.command.kpjoint * (self.leg_controller.command.qDes - self.leg_controller.data.q) + \
                self.leg_controller.command.kdjoint * (self.leg_controller.command.qdDes - self.leg_controller.data.qd)
        torque = (torque_ff + torque_fb).reshape(self.num_envs, -1) # (batch_size, 10)
        torque = torch.clamp(torque, -self.torque_limit, self.torque_limit)
        return torque