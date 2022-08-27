import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import numpy as np

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        self._rot = R.from_euler('zyx', [0.1, 0.0, 0.0], degrees=True)


    def rotate(self, rot):
        self._rot = rot


    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        vec = self._rot.apply(np.array([xs3d[1], ys3d[1], zs3d[1]]))
        xs, ys, zs = proj3d.proj_transform([xs3d[0], vec[0]], [ys3d[0], vec[1]], [zs3d[0], vec[2]], renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def update(frame):
    global yaw, pitch, roll
    global arrowX, arrowY, arrowZ
    global vecX, vecY, vecZ
    yaw_end = 45.0
    progress = yaw_end * frame/float(frames)
    arrowX.rotate(R.from_euler('zyx', [progress,0.0,0.0], degrees=True))
    return arrowX, 


def main():
    f = plt.figure(constrained_layout=True)
    f.suptitle("SO(3) Playground")
    gs = GridSpec(3, 3, figure=f)
    ax = f.add_subplot(gs[:, :], projection='3d')

    arrowRefX = Arrow3D([-1.0, 0.0], [-1.0, -1.0], [-1.0, -1.0],  mutation_scale=20, lw=3, arrowstyle="-|>", color="k")
    arrowRefY = Arrow3D([-1.0, -1.0], [-1.0, 0.0], [-1.0, -1.0],  mutation_scale=20, lw=3, arrowstyle="-|>", color="k")
    arrowRefZ = Arrow3D([-1.0, -1.0], [-1.0, -1.0], [-1.0, 0.0],  mutation_scale=20, lw=3, arrowstyle="-|>", color="k")
    ax.add_artist(arrowRefX)
    ax.add_artist(arrowRefY)
    ax.add_artist(arrowRefZ)

    global arrowX
    global arrowY
    global arrowZ
    global vecX
    global vecY
    global vecZ
    global frames
    frames = 300
    vecX = np.array([1.0, 0.0, 0.0])
    vecY = np.array([0.0, 1.0, 0.0])
    vecZ = np.array([0.0, 0.0, 1.0])
    arrowX = Arrow3D([0.0, vecX[0]], [0.0, vecX[1]], [0.0, vecX[2]],  mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    arrowY = Arrow3D([0.0, vecY[0]], [0.0, vecY[1]], [0.0, vecY[2]],  mutation_scale=20, lw=3, arrowstyle="-|>", color="g")
    arrowZ = Arrow3D([0.0, vecZ[0]], [0.0, vecZ[1]], [0.0, vecZ[2]],  mutation_scale=20, lw=3, arrowstyle="-|>", color="b")
    ax.add_artist(arrowX)
    ax.add_artist(arrowY)
    ax.add_artist(arrowZ)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.axis('off')

    anim = FuncAnimation(f, update, frames=frames, interval=10, blit=True)

    plt.show()


if __name__ == "__main__":
    main()
