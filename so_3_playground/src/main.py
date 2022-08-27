import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


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
    arrowX = Arrow3D([0.0, 1.0], [0.0, 0.0], [0.0, 0.0],  mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    arrowY = Arrow3D([0.0, 0.0], [0.0, 1.0], [0.0, 0.0],  mutation_scale=20, lw=3, arrowstyle="-|>", color="g")
    arrowZ = Arrow3D([0.0, 0.0], [0.0, 0.0], [0.0, 1.0],  mutation_scale=20, lw=3, arrowstyle="-|>", color="b")
    ax.add_artist(arrowX)
    ax.add_artist(arrowY)
    ax.add_artist(arrowZ)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.axis('off')   
    plt.show()


if __name__ == "__main__":
    main()
