import numpy as np


# define Cloud class as a prerequisite to load data
class Cloud:
    def __init__(self, points_path, position_path, subsetting=1):
        """
        :param position_path: file path of numpy file with tree positions: File must be of shape n_trees x 3
        :param points_path: file path of numpy file with point cloud. File must be of shape n_points x 4
        :param subsetting: filters the point cloud by the given rate. For 1 each point is selected
        """

        # load points and positions from path
        self.positions = np.load(position_path)
        self.points = np.load(points_path)
        self.filtered_points = None
        self.nbrs = None
        self.center = None

        # subset point cloud for better rendering performance
        self.points = self.points[::subsetting]

    def filter(self, center: list, radius: float, circle=True, remove999=False):
        assert radius > 0
        points = self.points
        # remove non tree points
        if remove999:
            index = (points[:, -1]).astype(int) != 999
            points = points[index]
        xypoints = points[:, [0, 1]]

        # calculate two norm or infinity norm of point distances to center
        if circle:
            distances = np.linalg.norm(xypoints - center[:2], ord=None, axis=1)
        else:
            distances = np.linalg.norm(xypoints - center[:2], ord=np.inf, axis=1)

        # check which points are within the radius
        within_radius = distances <= radius

        if not any(within_radius):
            raise EmptyFilter

        self.center = center
        self.filtered_points = points[within_radius]

    def plot_pptk(self, size=0.01):

        # Add tree position with scatter to filtered points to make center more visible in plot
        center = self.center
        center = np.append(center, 69)
        center = center.reshape(1,4) + np.random.uniform(low=0.0, high=0.1, size=(40,4))
        drawpoints = np.append(self.filtered_points, center, axis=0)

        import pptk

        # define a color palette
        np.random.seed(5)
        n_color_palette = len(np.unique(self.points[:, 3])) + 1
        color_palette = pptk.rand(n_color_palette, 3)

        color_palette[-1] = [0,0,0]

        # define color array by using label and color_palette
        num_drawpoints = len(drawpoints)
        colors = np.empty((num_drawpoints, 3))

        for i in range(num_drawpoints-40):
            ind = int(drawpoints[i][-1])

            if not ind == 999:
                colors[i] = color_palette[ind]
            else:
                colors[i] = color_palette[-2]

        colors[-40:] = np.repeat(np.array(color_palette)[-1].reshape(1,3), 40, 0)

        v = pptk.viewer(drawpoints[:, :-1])
        v.attributes(colors)
        v.set(point_size=size)
        v.set(lookat=drawpoints[-1, :-1])

    def plot_predictions(self, size=0.01):

        # Add tree position with scatter to filtered points to make center more visible in plot
        center = [0, 0, 0]
        center = np.append(center, 1)
        center = center.reshape(1,4) + np.random.uniform(low=0.0, high=0.1, size=(40,4))
        drawpoints = np.append(self.points, center, axis=0)

        import pptk

        num_drawpoints = len(drawpoints)
        colors = np.ones((num_drawpoints, 3))
        colors[:, :2] = colors[:, :2] * (1 - drawpoints[:, 3]).reshape(num_drawpoints, 1)

        colors[-40:] = np.repeat(np.array([0.4, 0.4, 0.4]).reshape(1,3), 40, 0)

        v = pptk.viewer(drawpoints[:, :-1])
        v.attributes(colors)
        v.set(point_size=size)
        v.set(lookat=drawpoints[-1, :-1])

    # todo: Not yet sure if we need this method
    def slice_up(self, slices: int):
        """
        The purpose of this method is to partition the point cloud along the z axis into a given number of slices
        :param slices:
        :return:
        """

        # sort point cloud by z coordinate in descending order
        self.points = self.points[self.points[:, 2].argsort()]
        points = self.points

        # determine breakpoints along the z coordinate
        minz = points[0][2]
        maxz = points[-1][2]
        breakpoints = np.linspace(start=minz, stop=maxz, num=slices)
        sliceindex = np.zeros(shape=breakpoints.shape, dtype=int)

        # associate indices with the breakpoints
        for i, breakposition in enumerate(breakpoints):
            sliceindex[i] = np.where(points[:, 2] >= breakposition)[0].min()

        return sliceindex

    # todo: Not yet sure if we need this method
    def find_neighbors(self, point, k: int):
        """
        returns the k nearest neighbors of point, plus point itself
        """
        # k neighbors plus point itself
        k = k + 1

        from sklearn.neighbors import NearestNeighbors

        if self.nbrs is None:
            self.nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(self.points[:,:-1])

        point = point.reshape(3, 1)
        _, indices = self.nbrs.kneighbors(point)
        return self.points[:,:-1][indices]


# Error class to catch if filter returns empty array
class EmptyFilter(Exception):
    def __str__(self):
        return "Filter does not contain any data"
        