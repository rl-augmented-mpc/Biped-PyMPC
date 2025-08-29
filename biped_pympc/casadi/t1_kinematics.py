import os
from typing import Optional, List
import casadi as cs
import pinocchio as pin
import pinocchio.casadi as cpin
from pathlib import Path


class T1FunctionGenerator:
    def __init__(self, urdf_path: str, package_dirs: Optional[List[str]] = None, cache_dir: str = "hector_pytorch/casadi/function"):
        self.urdf_path = urdf_path
        self.package_dirs = package_dirs or []
        self.cache_dir = cache_dir

        self.model = self._load_model()
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        self.nq = self.model.nq
        self.all_joint_names = [self.model.names[i] for i in range(1, self.model.njoints)]
        
    
    def _set_target_joints(self, side:str="Left"):
        # order preserved!
        self.joints_of_interest = [
            f'{side}_Hip_Pitch', f'{side}_Hip_Roll', f'{side}_Hip_Yaw',
            f'{side}_Knee_Pitch', f'{side}_Ankle_Pitch', f'{side}_Ankle_Roll',
        ]
        self.joint_names = [name for name in self.all_joint_names if name in self.joints_of_interest]
        self.joint_name_to_id = {name: i + 1 for i, name in enumerate(self.all_joint_names) if name in self.joints_of_interest}
        
        self.filtered_nq = len(self.joint_names)
        self.q_filtered_sym = cs.SX.sym('q_filtered', self.filtered_nq)

    def _load_model(self) -> pin.Model:
        if self.package_dirs:
            return pin.buildModelFromUrdf(self.urdf_path, pin.JointModelFreeFlyer())
        return pin.buildModelFromUrdf(self.urdf_path)

    def fk_func(self, frame_name: str) -> cs.Function:
        q = self.q_filtered_sym
        q_full = cs.SX.zeros(self.nq)
        for i, joint_name in enumerate(self.joint_names):
            joint_id = self.joint_name_to_id[joint_name]
            q_full[joint_id - 1] = q[i]

        cpin.forwardKinematics(self.cmodel, self.cdata, q_full)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        frame_id = next((i for i, f in enumerate(self.cmodel.frames) if f.name == frame_name), None)
        if frame_id is None:
            raise ValueError(f"Frame '{frame_name}' not found.")

        pos = self.cdata.oMf[frame_id].translation
        return cs.Function(f'fk_{frame_name}', [q], [pos])

    def jacobian_func(self, frame_name: str, reference_frame: pin.ReferenceFrame = pin.LOCAL_WORLD_ALIGNED) -> cs.Function:
        q = self.q_filtered_sym
        q_full = cs.SX.zeros(self.nq)
        for i, joint_name in enumerate(self.joint_names):
            joint_id = self.joint_name_to_id[joint_name]
            q_full[joint_id - 1] = q[i]

        cpin.forwardKinematics(self.cmodel, self.cdata, q_full)

        frame_id = next((i for i, f in enumerate(self.cmodel.frames) if f.name == frame_name), None)
        if frame_id is None:
            raise ValueError(f"Frame '{frame_name}' not found.")

        J_full = cpin.computeFrameJacobian(self.cmodel, self.cdata, q_full, frame_id, reference_frame)
        J_filtered = cs.SX.zeros(6, self.filtered_nq)
        for i, joint_name in enumerate(self.joint_names):
            joint_id = self.joint_name_to_id[joint_name]
            J_filtered[:, i] = J_full[:, joint_id - 1]

        return cs.Function(f'jacobian_{frame_name}', [q], [J_filtered])

    def save_functions(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # make left foot function
        self._set_target_joints(side="Left")
        functions = {
            "t1_fk_left": self.fk_func("left_foot_sole_link"),
            "t1_jac_left": self.jacobian_func("left_foot_sole_link"),
        }
        
        # make right foot function
        self._set_target_joints(side="Right")
        functions.update({
            "t1_fk_right": self.fk_func("right_foot_sole_link"),
            "t1_jac_right": self.jacobian_func("right_foot_sole_link"),
        })

        for name, func in functions.items():
            path = os.path.join(self.cache_dir, f"{name}.casadi")
            if not os.path.exists(path):
                func.save(path)
                print(f"Saved {name}.casadi")
            else:
                print(f"Skipped saving {name}, file already exists.")


if __name__ == "__main__":
    urdf_path = str(Path(__file__).parent.parent.parent / "model/t1_serial.urdf")
    cache_dir = str(Path(__file__).parent.parent / "casadi/function")
    print(f"Generating CasADi functions using URDF: {urdf_path}")
    generator = T1FunctionGenerator(urdf_path=urdf_path, cache_dir=cache_dir)
    generator.save_functions()
    print("Function generation complete.")