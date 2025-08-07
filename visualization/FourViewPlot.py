import matplotlib.pyplot as plt

class FourViewPlot:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 12))

        self.plotYZ = self.fig.add_subplot(2, 2, 1)
        self.plot3D = self.fig.add_subplot(2, 2, 2, projection='3d')
        self.plotXY = self.fig.add_subplot(2, 2, 3)
        self.plotXZ = self.fig.add_subplot(2, 2, 4)

    def addTrajectory(self, pos, name, color='b'):
        self.plotYZ.plot(pos[:,1], pos[:,2], c=color)
        self.plot3D.plot(*pos.T, label=name, c=color, alpha=0.75)
        self.plot3D.scatter(*pos[0,:], label=f'{name} Start', c=color, marker='^', s=75)
        self.plot3D.scatter(*pos[-1,:], label=f'{name} End', c=color, marker='*', s=75)
        self.plotXY.plot(pos[:,0], pos[:,1], c=color)
        self.plotXZ.plot(pos[:,0], pos[:,2], c=color)

    def addScatter(self, pos, name, color='b', marker='.'):
        self.plotYZ.scatter(pos[:,1], pos[:,2], label=f'{name}', c=color, marker=marker)
        self.plot3D.scatter(pos[:,0], pos[:,1], pos[:,2], label=f'{name}', c=color, marker=marker)
        self.plotXY.scatter(pos[:,0], pos[:,1], label=f'{name}', c=color, marker=marker)
        self.plotXZ.scatter(pos[:,0], pos[:,2], label=f'{name}', c=color, marker=marker)

    def show(self):
        self.plotYZ.set_title('Y-Z')
        self.plotYZ.set_xlabel('Y')
        self.plotYZ.set_ylabel('Z')
        self.plotYZ.grid()
        self.plotYZ.axis('equal')

        self.plot3D.set_title('Vehicle Trajectory')
        self.plot3D.set_xlabel('X Position')
        self.plot3D.set_ylabel('Y Position')
        self.plot3D.set_zlabel('Z Position')
        self.plot3D.grid()
        self.plot3D.axis('equal')
        self.plot3D.legend()

        self.plotXY.set_title('X-Y')
        self.plotXY.set_xlabel('X')
        self.plotXY.set_ylabel('Y')
        self.plotXY.grid()
        self.plotXY.axis('equal')

        self.plotXZ.set_title('X-Z')
        self.plotXZ.set_xlabel('X')
        self.plotXZ.set_ylabel('Z')
        self.plotXZ.grid()
        self.plotXZ.axis('equal')

        self.fig.subplots_adjust(
            left=0.06,
            bottom=0.04,
            right=0.96,
            top=0.97,
            wspace=0.14,
            hspace=0.138
        )

        plt.show()