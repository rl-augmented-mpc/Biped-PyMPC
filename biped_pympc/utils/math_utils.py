import torch

@torch.jit.script
def rot_x(angle: torch.Tensor) -> torch.Tensor:
    """
    Batched rotation matrix around x-axis.
    Input: angle of shape (B,)
    Output: rotation matrices of shape (B, 3, 3)
    """
    c = torch.cos(angle)
    s = torch.sin(angle)
    ones = torch.ones_like(angle)
    zeros = torch.zeros_like(angle)

    row1 = torch.stack([ones, zeros, zeros], dim=-1)   # shape (B, 3)
    row2 = torch.stack([zeros, c, -s], dim=-1)           # shape (B, 3)
    row3 = torch.stack([zeros, s, c], dim=-1)            # shape (B, 3)
    return torch.stack([row1, row2, row3], dim=-2)        # shape (B, 3, 3)

@torch.jit.script
def rot_y(angle: torch.Tensor) -> torch.Tensor:
    """
    Batched rotation matrix around y-axis.
    """
    c = torch.cos(angle)
    s = torch.sin(angle)
    ones = torch.ones_like(angle)
    zeros = torch.zeros_like(angle)

    row1 = torch.stack([c, zeros, s], dim=-1)
    row2 = torch.stack([zeros, ones, zeros], dim=-1)
    row3 = torch.stack([-s, zeros, c], dim=-1)
    
    return torch.stack([row1, row2, row3], dim=-2)

# @torch.jit.script
def rot_z(angle: torch.Tensor) -> torch.Tensor:
    """
    Batched rotation matrix around z-axis.
    """
    c = torch.cos(angle)
    s = torch.sin(angle)
    ones = torch.ones_like(angle)
    zeros = torch.zeros_like(angle)

    row1 = torch.stack([c, -s, zeros], dim=-1)
    row2 = torch.stack([s, c, zeros], dim=-1)
    row3 = torch.stack([zeros, zeros, ones], dim=-1)
    
    return torch.stack([row1, row2, row3], dim=-2)

@torch.jit.script
def quaternion_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of quaternions to their corresponding 3x3 rotation matrices.

    Args:
        quat (torch.Tensor): A tensor of shape (batch_size, 4),
                             where each quaternion is in the form (w, x, y, z).

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 3, 3) containing the
                      rotation matrices corresponding to the input quaternions.
    """
    # Ensure input is a float tensor
    quat = quat.float()

    # Normalize the quaternions to unit length to avoid scaling in the rotation
    norm_quat = torch.norm(quat, p=2, dim=1, keepdim=True)
    quat = quat / norm_quat

    # Unpack the quaternions
    w, x, y, z = quat.unbind(dim=1)

    # Compute elements of the rotation matrix
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    xy = x * y
    xz = x * z
    yz = y * z

    wx = w * x
    wy = w * y
    wz = w * z

    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)

    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)

    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz

    # Stack into rotation matrices
    rotation_matrix = torch.stack([
        torch.stack([r00, r01, r02], dim=1),
        torch.stack([r10, r11, r12], dim=1),
        torch.stack([r20, r21, r22], dim=1)
    ], dim=1)

    return rotation_matrix

@torch.jit.script
def quat_to_euler(quat):
    """
    Convert a batch of quaternions to Euler angles (roll, pitch, yaw) in radians.

    Args:
        quat (torch.Tensor): Tensor of shape (B, 4) representing quaternions in [w, x, y, z] format.
                             If you have a single quaternion, ensure it has shape (1, 4) or adjust accordingly.

    Returns:
        torch.Tensor: Tensor of shape (B, 3) containing the Euler angles (roll, pitch, yaw) for each quaternion.
    """
    # Ensure the quaternion is of floating point type.
    quat = quat.float()

    # Unpack the quaternion components
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Roll (rotation around the x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))

    # Pitch (rotation around the y-axis)
    sinp = 2.0 * (w * y - z * x)
    # Clamp sinp to the interval [-1, 1] to avoid numerical errors with asin
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))

    # Yaw (rotation around the z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

    # Stack the angles into a tensor of shape (B, 3)
    return torch.stack([roll, pitch, yaw], dim=-1)
    


@torch.jit.script
def skew_symmetric(foot_position):
    """
    Converts foot positions (B, 3) into skew-symmetric ones (B, 3, 3)
    """
    batch_size = foot_position.shape[0]
    x = foot_position[:, 0]
    y = foot_position[:, 1]
    z = foot_position[:, 2]
    zero = torch.zeros_like(x)
    skew = torch.stack([
        zero, -z, y, 
        z, zero, -x, 
        -y, x, zero], dim=1).reshape(
        (batch_size, 3, 3))
    return skew

@torch.jit.script
def unskew_symmetric(skew):
    """
    Converts skew-symmetric (B, T, 3, 3) into foot positions (B, T, 3,)
    """
    x = skew[:, :, 2, 1]
    y = skew[:, :, 0, 2]
    z = skew[:, :, 1, 0]
    return torch.stack([x, y, z], dim=2)