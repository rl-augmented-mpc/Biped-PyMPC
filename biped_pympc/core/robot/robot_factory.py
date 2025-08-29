from biped_pympc.core.robot.hector import HECTOR
from biped_pympc.core.robot.t1 import T1

class RobotFactory:
    def __init__(self, robot_type: str):
        self.robot_type = robot_type.lower()
        
    def __call__(self, *args, **kwargs):
        if self.robot_type == "hector":
            return HECTOR(*args, **kwargs)
        elif self.robot_type == "t1":
            return T1(*args, **kwargs)
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}. Supported types are 'hector' and 't1'.")