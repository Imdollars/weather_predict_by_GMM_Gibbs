import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import wishart
from sklearn.preprocessing import LabelEncoder, StandardScaler


def gibbs_sampling_gmm(data, k, iterations):

    # initialize
    n, d = data.shape
    alpha = np.ones(k)
    beta = 1.0
    mu = np.random.rand(k, d)
    sigma = np.array([np.eye(d) for _ in range(k)])
    z = np.random.randint(0, k, size=n)
    pi = np.random.dirichlet(alpha)
    eta = np.zeros((n, k))
    w = np.eye(d)
    nu = d

    # initialize visualizing objects
    snapshots = []

    # iteration start
    for iteration in range(iterations):

        # sampling z
        for i in range(n):
            for j in range(k):
                eta[i, j] = np.exp(-0.5 * np.dot((data[i] - mu[j]).T, np.linalg.solve(sigma[j], (data[i] - mu[j])))
                                   + 0.5 * np.log(np.linalg.det(np.linalg.inv(sigma[j])))
                                   + np.log(pi[j]))
            eta_sum = eta[i, :].sum()
            if eta_sum > 0:
                eta[i, :] /= eta_sum
            else:
                eta[i, :] = 1.0 / k  # assign equal probability if sum is zero for avoiding errors
            z[i] = np.random.choice(k, p=eta[i, :])

        # sampling sigma and mu
        for j in range(k):
            points = data[z == j]
            n_j = len(points)
            if n_j > 0:
                x_bar = points.mean(axis=0)
                S_j = np.dot((points - x_bar).T, (points - x_bar))
                beta_k = beta + n_j
                mu[j] = (beta * mu[j] + n_j * x_bar) / beta_k
                W_k = np.linalg.inv(
                    np.linalg.inv(w) + S_j + (beta * n_j / beta_k) * np.outer(x_bar - mu[j], x_bar - mu[j]))
                sigma[j] = wishart(df=nu + n_j, scale=W_k).rvs()
            else:
                mu[j] = np.random.rand(d)
                sigma[j] = np.eye(d)

        # sampling pi
        alpha_k = alpha + np.array([np.sum(z == j) for j in range(k)])
        pi = np.random.dirichlet(alpha_k)

        #
        snapshots.append((z.copy(), mu.copy()))

    return z, mu, pi, sigma, snapshots


def create_3d_animation(data, snapshots, k, iterations, filename='gibbs_sampling_gmm_3d.gif'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # colours extended list

    def update(frame):
        ax.clear()
        z, mu = snapshots[frame]
        for i in range(k):
            points = data[z == i]
            if len(points) > 0:  # Check if there are points in the cluster
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors[i], label=f'Cluster {i}')
                ax.scatter(mu[i][0], mu[i][1], mu[i][2], c=colors[i], marker='x', s=200, linewidths=3)

        # Calculate zoom level
        # zoom_factor = 1 + frame * 0.02  # Adjust the zoom factor as needed
        # center = np.mean(data, axis=0)
        # range_x = (data[:, 0].max() - data[:, 0].min()) / zoom_factor
        # range_y = (data[:, 1].max() - data[:, 1].min()) / zoom_factor
        # range_z = (data[:, 2].max() - data[:, 2].min()) / zoom_factor

        center = np.mean(data, axis=0)
        range_x = (data[:, 0].max() - data[:, 0].min())
        range_y = (data[:, 1].max() - data[:, 1].min())
        range_z = (data[:, 2].max() - data[:, 2].min()) 

        ax.set_xlim(center[0] - range_x / 2, center[0] + range_x / 2)
        ax.set_ylim(center[1] - range_y / 2, center[1] + range_y / 2)
        ax.set_zlim(center[2] - range_z / 2, center[2] + range_z / 2)

        ax.set_title('Gibbs Sampling GMM Clustering')
        ax.legend()
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')

    ani = FuncAnimation(fig, update, frames=iterations, repeat=False)
    writer = PillowWriter(fps=5)
    ani.save(filename, writer=writer)
    plt.show()


if __name__ == "__main__":
    # Load the data from the file
    file_path = 'weather_data.csv'
    data = pd.read_csv(file_path)

    # Inspect the first few rows of the data to understand its structure
    print(data.head())
    print(data.columns)

    # Replace string columns with numerical columns using Label Encoding
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            data[column] = label_encoders[column].fit_transform(data[column])

    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    points = data_normalized[:, :]

    print(data.head())
    print(data.columns)
    print(data_normalized)

    # Apply Gibbs sampling
    k = 4
    iterations = 500
    z, mu, pi, sigma, snapshots = gibbs_sampling_gmm(points, k, iterations)

    file_path_original = 'weather_classification_data.csv'
    data_original = pd.read_csv(file_path_original)
    label_encoders = {}
    for column in data_original.columns:
        if data_original[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            data_original[column] = label_encoders[column].fit_transform(data_original[column])

    weather_type = data_original['Weather Type'].tolist()
    count1 = np.zeros(k)
    for i in range(len(z)):
        for j in range(k):
            if z[i] == j:
                count1[j] += 1

    count2 = np.zeros(k)
    for i in range(len(weather_type)):
        for j in range(k):
            if weather_type[i] == j:
                count2[j] += 1

    # print(data_original)
    # print(weather_type)
    print(count1)
    print(count2)
    print(mu)
    print(pi)
    print(sigma)

    # Create and save the animation
    for i in range(7):
        create_3d_animation(points[:, i:(i+3)], snapshots, k, iterations)
