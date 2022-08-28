import rlcompleter
from timeit import repeat
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import numpy as np
import numpy.linalg

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        self._rot = R.from_euler('ZYX', [0.1, 0.0, 0.0], degrees=True)


    def rotate(self, rot):
        self._rot = rot


    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        vec = self._rot.apply(np.array([xs3d[1], ys3d[1], zs3d[1]]))
        xs, ys, zs = proj3d.proj_transform([xs3d[0], vec[0]], [ys3d[0], vec[1]], [zs3d[0], vec[2]], renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def update(frame):
    # update euler plot
    global yaw, pitch, roll
    global progress_yaw, progress_pitch, progress_roll
    global arrowXEuler, arrowYEuler, arrowZEuler
    third_of_frames=int(frames/3.0)
    if frame < third_of_frames:
        progress_yaw = yaw * float(frame+1)/third_of_frames
    elif frame < (2 * third_of_frames):
        progress_pitch = pitch * (float(frame+1)/third_of_frames-1.0)
    else:
        progress_roll = roll * (float(frame+1)/third_of_frames-2.0)
    euler = R.from_euler('ZYX', [progress_yaw, progress_pitch, progress_roll], degrees=True)
    arrowXEuler.rotate(euler)
    arrowYEuler.rotate(euler)
    arrowZEuler.rotate(euler)
    
    # update rot-vec plot
    global angle, rotvec_norm
    global angle_progress
    angle_progress = angle * float(frame+1)/float(frames)
    rot_vec = R.from_rotvec(np.deg2rad(angle_progress)*rotvec_norm)
    arrowXRotVec.rotate(rot_vec)
    arrowYRotVec.rotate(rot_vec)
    arrowZRotVec.rotate(rot_vec)

    return (arrowXEuler, arrowYEuler, arrowZEuler, arrowXRotVec, arrowYRotVec, arrowZRotVec) 


def generate_euler_plot(ax):
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.axis('off')

    arrowRefX = Arrow3D([-1.0, 0.0], [-1.0, -1.0], [-1.0, -1.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="k")
    arrowRefY = Arrow3D([-1.0, -1.0], [-1.0, 0.0], [-1.0, -1.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="k")
    arrowRefZ = Arrow3D([-1.0, -1.0], [-1.0, -1.0], [-1.0, 0.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="k")
    ax.add_artist(arrowRefX)
    ax.add_artist(arrowRefY)
    ax.add_artist(arrowRefZ)
    
    arrowTargetX = Arrow3D([0.0, 1.0], [0.0, 0.0], [0.0,  0.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="r", alpha=0.1)
    arrowTargetX.rotate(R.from_euler('ZYX', [yaw, pitch, roll], degrees=True))
    arrowTargetY = Arrow3D([0.0, 0.0], [0.0, 1.0], [0.0, 0.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="g", alpha=0.1)
    arrowTargetY.rotate(R.from_euler('ZYX', [yaw, pitch, roll], degrees=True))
    arrowTargetZ = Arrow3D([0.0, 0.0], [0.0, 0.0], [0.0, 1.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="b", alpha=0.1)
    arrowTargetZ.rotate(R.from_euler('ZYX', [yaw, pitch, roll], degrees=True))
    ax.add_artist(arrowTargetX)
    ax.add_artist(arrowTargetY)
    ax.add_artist(arrowTargetZ)

    global arrowXEuler
    global arrowYEuler
    global arrowZEuler
    arrowXEuler = Arrow3D([0.0, 1.0], [0.0, 0.0], [0.0,  0.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="r")
    arrowYEuler = Arrow3D([0.0, 0.0], [0.0, 1.0], [0.0, 0.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="g")
    arrowZEuler = Arrow3D([0.0, 0.0], [0.0, 0.0], [0.0, 1.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="b")
    ax.add_artist(arrowXEuler)
    ax.add_artist(arrowYEuler)
    ax.add_artist(arrowZEuler)


def generate_rot_vec(ax):
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.axis('off')

    arrowRefX = Arrow3D([-1.0, 0.0], [-1.0, -1.0], [-1.0, -1.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="k")
    arrowRefY = Arrow3D([-1.0, -1.0], [-1.0, 0.0], [-1.0, -1.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="k")
    arrowRefZ = Arrow3D([-1.0, -1.0], [-1.0, -1.0], [-1.0, 0.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="k")
    ax.add_artist(arrowRefX)
    ax.add_artist(arrowRefY)
    ax.add_artist(arrowRefZ)
    
    arrowTargetX = Arrow3D([0.0, 1.0], [0.0, 0.0], [0.0,  0.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="r", alpha=0.1)
    arrowTargetX.rotate(R.from_euler('ZYX', [yaw, pitch, roll], degrees=True))
    arrowTargetY = Arrow3D([0.0, 0.0], [0.0, 1.0], [0.0, 0.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="g", alpha=0.1)
    arrowTargetY.rotate(R.from_euler('ZYX', [yaw, pitch, roll], degrees=True))
    arrowTargetZ = Arrow3D([0.0, 0.0], [0.0, 0.0], [0.0, 1.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="b", alpha=0.1)
    arrowTargetZ.rotate(R.from_euler('ZYX', [yaw, pitch, roll], degrees=True))
    ax.add_artist(arrowTargetX)
    ax.add_artist(arrowTargetY)
    ax.add_artist(arrowTargetZ)

    global arrowXRotVec
    global arrowYRotVec
    global arrowZRotVec
    arrowXRotVec = Arrow3D([0.0, 1.0], [0.0, 0.0], [0.0,  0.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="r")
    arrowYRotVec = Arrow3D([0.0, 0.0], [0.0, 1.0], [0.0, 0.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="g")
    arrowZRotVec = Arrow3D([0.0, 0.0], [0.0, 0.0], [0.0, 1.0],  mutation_scale=20, lw=2, arrowstyle="-|>", color="b")
    ax.add_artist(arrowXRotVec)
    ax.add_artist(arrowYRotVec)
    ax.add_artist(arrowZRotVec)


def main():
    global yaw, pitch, roll
    yaw = 25.0
    pitch = -45.0
    roll = 45.0
    global progress_yaw, progress_pitch, progress_roll
    progress_yaw = 0.0
    progress_pitch = 0.0
    progress_roll = 0.0
    global angle, rotvec_norm
    rotvec_norm = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_rotvec()
    rotvec_norm = 1.0/np.linalg.norm(R.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_rotvec()) * rotvec_norm
    angle = np.rad2deg(np.linalg.norm(R.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_rotvec()))
    global angle_progress
    angle_progress = 0.0

    f = plt.figure(constrained_layout=True, figsize=(16,9))
    f.suptitle("SO(3) Playground")
    gs = GridSpec(2, 2, figure=f)

    ax1 = f.add_subplot(gs[0, 0], projection='3d')
    generate_euler_plot(ax1)
    ax2 = f.add_subplot(gs[0, 1], projection='3d')
    generate_rot_vec(ax2)

    global frames
    frames = 300
    anim = FuncAnimation(f, update, frames=frames, interval=30, blit=True, repeat=False)

    plt.show()


if __name__ == "__main__":
    main()
