import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

def animate_srbd(
    poses, 
    forces_L, forces_R, 
    moments_L, moments_R,
    foot_L=(0.1, 0.05, 0.0), 
    foot_R=(0.1, -0.05, 0.0),
    l=0.2, w=0.1, h=0.05, 
    sphere_r=0.03, sphere_res=(10, 20),
    interval=50, 
    force_scale=0.1, 
    moment_scale=0.1
):
    """
    Animate a 3D box with spherical foot contacts and overlay force/moment vectors.

    Parameters:
    -----------
    poses       : array-like, shape (T,6)   rows = [roll, pitch, yaw, x, y, z]
    forces_L    : array-like, shape (T,3)   left foot force vectors
    forces_R    : array-like, shape (T,3)   right foot force vectors
    moments_L   : array-like, shape (T,3)   left foot moment vectors
    moments_R   : array-like, shape (T,3)   right foot moment vectors
    foot_L      : tuple of 3 floats         left foot pos in body frame
    foot_R      : tuple of 3 floats         right foot pos in body frame
    l, w, h     : floats                    box dimensions
    sphere_r    : float                     sphere radius
    sphere_res  : tuple (n_theta, n_phi)    sphere resolution
    interval    : int                       ms between frames
    force_scale : float                     scale length of force arrows
    moment_scale: float                     scale length of moment arrows
    """
    poses     = np.asarray(poses)
    forces_L  = np.asarray(forces_L)
    forces_R  = np.asarray(forces_R)
    moments_L = np.asarray(moments_L)
    moments_R = np.asarray(moments_R)
    T = poses.shape[0]

    # Box corners and faces
    corners = np.array([
        [-l/2, -w/2, -h/2], [ l/2, -w/2, -h/2],
        [ l/2,  w/2, -h/2], [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2], [ l/2, -w/2,  h/2],
        [ l/2,  w/2,  h/2], [-l/2,  w/2,  h/2],
    ])
    faces_idx = [
        [0,1,2,3], [4,5,6,7],
        [0,1,5,4], [2,3,7,6],
        [1,2,6,5], [0,3,7,4],
    ]

    # Prepare unit sphere points
    n_theta, n_phi = sphere_res
    theta = np.linspace(0, np.pi, n_theta)
    phi   = np.linspace(0, 2*np.pi, n_phi)
    Th, Ph = np.meshgrid(theta, phi)
    Xs = sphere_r * np.sin(Th) * np.cos(Ph)
    Ys = sphere_r * np.sin(Th) * np.sin(Ph)
    Zs = sphere_r * np.cos(Th)
    sphere_pts = np.vstack([Xs.ravel(), Ys.ravel(), Zs.ravel()]).T

    # Euler ZYX → Rotation
    def euler_to_R(r, p, y):
        Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
        Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
        Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
        return Rz @ Ry @ Rx

    # Setup figure
    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(0,1)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    # Box patch
    box_surf = Poly3DCollection([], facecolors='green', edgecolors='k', alpha=0.7)
    ax.add_collection3d(box_surf)

    # Placeholders for primitives
    scatL = scatR = None
    qfL = qfR = qmL = qmR = None

    def update(i):
        nonlocal scatL, scatR, qfL, qfR, qmL, qmR

        # Remove previous artists
        if scatL: scatL.remove()
        if scatR: scatR.remove()
        if qfL: qfL.remove()
        if qfR: qfR.remove()
        if qmL: qmL.remove()
        if qmR: qmR.remove()

        # Get pose & rotation
        r, p, y, x0, y0, z0 = poses[i]
        R = euler_to_R(r, p, y)
        Tvec = np.array([x0, y0, z0])

        # Update box
        pts = (R @ corners.T).T + Tvec
        faces = [[pts[idx] for idx in face] for face in faces_idx]
        box_surf.set_verts(faces)

        # Foot positions in world frame
        fLw = np.array(foot_L)
        fRw = np.array(foot_R)
        # fLw = R @ fLb + Tvec
        # fRw = R @ fRb + Tvec

        # Plot spheres at feet
        sphL = (R @ sphere_pts.T).T + fLw
        sphR = (R @ sphere_pts.T).T + fRw
        scatL = ax.scatter(sphL[:,0], sphL[:,1], sphL[:,2], color='blue', s=10)
        scatR = ax.scatter(sphR[:,0], sphR[:,1], sphR[:,2], color='blue', s=10)

        # Plot forces (red) and moments (green)
        vecL = forces_L[i]; vecR = forces_R[i]
        qfL  = ax.quiver(*fLw, *vecL, length=force_scale, normalize=False, color='r')
        qfR  = ax.quiver(*fRw, *vecR, length=force_scale, normalize=False, color='r')
        # qmL  = ax.quiver(*fLw, *moments_L[i], length=moment_scale, normalize=False, color='g')
        # qmR  = ax.quiver(*fRw, *moments_R[i], length=moment_scale, normalize=False, color='g')

        return (box_surf, scatL, scatR, qfL, qfR, qmL, qmR)

    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    plt.show()
    return anim
