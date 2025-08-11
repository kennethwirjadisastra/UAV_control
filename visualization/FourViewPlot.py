import matplotlib.pyplot as plt

# plt.ion()

class FourViewPlot:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 12))

        self.plotYZ = self.fig.add_subplot(2, 2, 1)
        self.plot3D = self.fig.add_subplot(2, 2, 2, projection='3d')
        self.plotXY = self.fig.add_subplot(2, 2, 3)
        self.plotXZ = self.fig.add_subplot(2, 2, 4)

        self.curves = {}
        self.scatters = {}

    def addTrajectory(self, pos, name, color='b'):
        if name not in self.curves:
            CYZ,    = self.plotYZ.plot(pos[:,1], pos[:,2], c=color)
            C3D,    = self.plot3D.plot(*pos.T, label=name, c=color, alpha=0.75)
            C3DS    = self.plot3D.scatter(*pos[0,:], label=f'{name} Start', c=color, marker='^', s=75)
            C3DE    = self.plot3D.scatter(*pos[-1,:], label=f'{name} End', c=color, marker='*', s=75)
            CXY,    = self.plotXY.plot(pos[:,0], pos[:,1], c=color)
            CXZ,    = self.plotXZ.plot(pos[:,0], pos[:,2], c=color)
            curve   = [CYZ, C3D, C3DS, C3DE, CXY, CXZ]
            self.curves[name] = curve
        else:
            CYZ, C3D, C3DS, C3DE, CXY, CXZ = self.curves[name]
            CYZ.set_data(pos[:,1], pos[:,2])
            CXY.set_data(pos[:,0], pos[:,1])
            CXZ.set_data(pos[:,0], pos[:,2])
            C3D.set_data(pos[:,0], pos[:,1])
            C3D.set_3d_properties(pos[:,2])
            C3DS._offsets3d = (pos[0:1,0], pos[0:1,1], pos[0:1,2])
            C3DE._offsets3d = (pos[-1:,0], pos[-1:,1], pos[-1:,2])

    def addScatter(self, pos, name, color='b', marker='.'):
        if name not in self.scatters:
            SYZ = self.plotYZ.scatter(pos[:,1], pos[:,2], label=f'{name}', c=color, marker=marker)
            S3D = self.plot3D.scatter(pos[:,0], pos[:,1], pos[:,2], label=f'{name}', c=color, marker=marker)
            SXY = self.plotXY.scatter(pos[:,0], pos[:,1], label=f'{name}', c=color, marker=marker)
            SXZ = self.plotXZ.scatter(pos[:,0], pos[:,2], label=f'{name}', c=color, marker=marker)
            scatter = [SYZ, S3D, SXY, SXZ]
            self.scatters[name] = scatter
        else:
            SYZ, S3D, SXY, SXZ = self.scatters[name]
            SYZ.set_offsets(pos[:, [1, 2]])
            S3D._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
            SXY.set_offsets(pos[:, [0, 1]])
            SXZ.set_offsets(pos[:, [0, 2]])
        

    def show(self):
        plt.ion()
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


        self.fig.canvas.draw()       # draw initial plot
        self.fig.canvas.flush_events()  # process GUI events, show window
        plt.show()
        plt.ioff()
