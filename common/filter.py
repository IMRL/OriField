import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_covariance_ellipse(x, P, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    绘制协方差椭圆
    :param x: 状态向量
    :param P: 协方差矩阵
    :param ax: Matplotlib 轴对象
    :param n_std: 标准差的数量
    :param facecolor: 椭圆的填充颜色
    :param kwargs: 其他绘图参数
    """
    pos = x[:2]
    cov = P[:2, :2]
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle,
                      facecolor=facecolor, **kwargs)
    ax.add_patch(ellipse)
    return ellipse

def kf_vis(observations, filtered_x, filtered_P):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.plot(observations[:, 0], observations[:, 1], c='g')
    ax.plot(filtered_x[:, 0], filtered_x[:, 1], c='r')
    for i, (x, P) in enumerate(zip(filtered_x, filtered_P)):
        plot_covariance_ellipse(x, P, ax, edgecolor='r')
        ax.annotate(f"{i}", (observations[i, 0], observations[i, 1]), c="g")
        ax.annotate(f"{i}", (x[0], x[1]), c="r")
        print(f"Updated state estimate: i {i} x {x}")
    plt.gca().set_aspect('equal')
    plt.show()

class KalmanFilterNumpy:
    def __init__(self, F, H, Q, R, P, x):
        """
        初始化卡尔曼滤波器
        :param F: 状态转移矩阵
        :param H: 观测矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 观测噪声协方差矩阵
        :param P: 估计误差协方差矩阵
        :param x: 初始状态
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def predict(self):
        """
        预测步骤
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        """
        更新步骤
        :param z: 观测值
        """
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.F.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

def kalman_filter_numpy(observations, vis=False):
    F = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    Q = np.eye(4)*0.1
    
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    R = np.eye(2)*5

    x = np.array([0, 0, 0, 0])
    P = np.eye(4)#*1000  # Large initial uncertainty
    
    kf = KalmanFilterNumpy(F, H, Q, R, P, x)
    kf.x[:2] = observations[0]

    filtered_x = []
    filtered_P = []
    for z in observations:
        kf.predict()
        kf.update(z)
        filtered_x.append(kf.x[:2])
        filtered_P.append(kf.P[:2, :2])
    filtered_x = np.array(filtered_x)
    filtered_P = np.array(filtered_P)

    if vis:
        kf_vis(observations, filtered_x, filtered_P)

    return filtered_x, filtered_P

def kalman_filter(observations, vis=False):
    F = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    Q = np.eye(4)*0.1
    
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    R = np.eye(2)*5

    x = np.array([0, 0, 0, 0])
    P = np.eye(4)#*1000  # Large initial uncertainty

    # Initialize Kalman Filter
    kf = KalmanFilter(dim_x=4, dim_z=2)
    # State Transition Matrix
    kf.F = F
    # Measurement Matrix
    kf.H = H
    # Initial State Estimate
    kf.x = x
    # Process Noise Covariance
    kf.Q = Q
    # Measurement Noise Covariance
    kf.R = R
    # Initial Covariance Estimate
    kf.P = P
    kf.x[:2] = observations[0]

    # Store the filtered positions
    filtered_x = []
    filtered_P = []
    for z in observations:
        # Prediction Step
        kf.predict()
        # Update Step
        kf.update(z)
        # Store current position estimate
        filtered_x.append(kf.x[:2])
        filtered_P.append(kf.P[:2, :2])
    # Convert to numpy array for easy plotting
    filtered_x = np.array(filtered_x)
    filtered_P = np.array(filtered_P)

    if vis:
        kf_vis(observations, filtered_x, filtered_P)

    return filtered_x, filtered_P

class BayesianFusion:
    def __init__(self, initial_state):
        self.H = np.array([[1, 0],
                            [0, 1]])  # Measurement matrix
        self.R = np.eye(2) * 2  # Measurement noise
        self.x = initial_state
        self.P = np.eye(2) * 10  # Initial uncertainty

    def update(self, z):
        
        # Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        
        # Update error covariance
        I = np.eye(self.H.shape[1])
        self.P = (I - K @ self.H) @ self.P

def bayesian_fusion(observations, vis=False):
    # Example usage
    initial_state = observations[0]  # Initial guess for [x, y, w, h]
    fusion = BayesianFusion(initial_state)

    filtered_x = []
    filtered_P = []
    for z in observations:
        fusion.update(z)
        filtered_x.append(fusion.x[:2])
        filtered_P.append(fusion.P[:2, :2])
    filtered_x = np.array(filtered_x)
    filtered_P = np.array(filtered_P)

    if vis:
        kf_vis(observations, filtered_x, filtered_P)

    return filtered_x, filtered_P

if __name__ == "__main__":
    measurements = np.array([np.array([1, 1]), np.array([2, 2]), np.array([3, 3])])

    kalman_filter(measurements)