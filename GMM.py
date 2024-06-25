import numpy as np
from scipy.stats import multivariate_normal


# 初始化参数
def ini(data, k):  # x：数据矩阵， k：分布总数
    type = [0, 1, 2]    # 隐变量z值域
    n = data.shape[0]  # 数据量 （ data的列数 ）
    d = data.shape[1]  # 数据维度 （ data的行数 ）

    mu = np.random.rand(k, d)   # 平均矩阵 mu[k][d]
    pi = np.ones(k) / k         # 分布贡献度向量 pi[k]
    sigma = np.array([np.eye(d) for _ in range(k)])     # k个协方差矩阵 sigma[d][d]组成的数组

    z_possibility_orignal = np.random.rand(n, k)
    z_possibility = z_possibility_orignal / z_possibility_orignal.sum(1, keepdims=True)

    z = np.zeros(n)     # 隐变量向量 z[n]
    for i in range(n):
        z[i] = np.random.choice(type, size=None, p=z_possibility[i, :])


    return z, mu, pi, sigma, type, z_possibility


# 计算 x[n] 的隐变量 z[n] 的概率矩阵 responsibilities[n][k]
def z_possibility(data, mu, sigma, pi, k, z_possibility):
    for j in range(k):
        z_possibility[:, j] = pi[j] * multivariate_normal.pdf(data, mean=mu[j], cov=sigma[j])
    z_possibility = z_possibility / z_possibility.sum(1, keepdims=True)
    return z_possibility


def update_z(z_possibility):
    type = [0, 1, 2]
    n, k = z_possibility.shape   # 行：n，列：k
    z = np.zeros(n)
    for i in range(n):
            z[i] = np.random.choice(type, size=None, p=z_possibility[i, :])
    # 抽样更新 隐变量z[n]
    return z


def update_parameters(data, z, k, mu, sigma, pi):
    n, d = data.shape
    n_k = np.zeros(k)   # 第k个高斯分布的数据点量矩阵 n_k[k]
    x_k_sum = np.zeros(k, d)
    sigma_k_sum = np.array([np.zeros(d) for _ in range(k)])
    for j in range(k):
    #     x_k = data[z == j]
    #     print("x[z == j]", data[z == j])
    #     print("x_k", x_k)
    #     n_k = x_k.shape[0]
    #     print("n_k", n_k)
    #     if n_k > 0:
    #         mu[j] = x_k.mean(axis=0)
    #         sigma[j] = np.cov(x_k, rowvar=False) if n_k > 1 else np.eye(d)
    #         pi[j] = n_k / n
    #     else:
    #         mu[j] = np.random.rand(d)
    #         sigma[j] = np.eye(d)
    #         pi[j] = 1 / n
        for i in range(n):
            if z[i] == j:
                n_k[j] = n_k[j] + 1
                x_k_sum[j, :] += data[i, :]
                sigma_k_sum[j] += np.outer(data[i, :] - mu[j, :], data[i, :] - mu[j, :])
        pi[j] = n_k[j] / n
        sigma[j] = sigma_k_sum[j] / n_k[j]
        mu[j] = x_k_sum[j] / n_k[j]

    return mu, sigma, pi


def is_converged(mu_old, mu_new, threshold):
    return np.linalg.norm(mu_old - mu_new) < threshold


def gmm_gibbs(data, k, step, threshold):
    z, mu, pi, sigma, type, z_possibility = ini(data, k)
    # print("z", z)
    # print("mu", mu)
    # print("sigma", sigma)
    # print("pi", pi)
    for i in range(step):
        mu_old = mu.copy()
        z = update_z(z_possibility(data, mu, sigma, pi, k))
        mu, sigma, pi = update_parameters(data, z, k, mu, sigma, pi)
        # print("z", z)
        # print("mu", mu)
        # print("sigma", sigma)
        # print("pi", pi)
        if is_converged(mu_old, mu, threshold):
            break

    return z, mu, pi, sigma
    

if __name__ == '__main__':
    # np.random.seed(80)
    # sigma_1 = np.random.rand(4, 4)
    # while np.all(sigma_1 <= 0):
    #     sigma_1 = np.random.rand(4, 4)
    #
    # sigma_2 = np.random.rand(4, 4)
    # while np.all(sigma_2 <= 0):
    #     sigma_2 = np.random.rand(4, 4)
    #
    # sigma_3 = np.random.rand(4, 4)
    # while np.all(sigma_3 <= 0):
    #     sigma_3 = np.random.rand(4, 4)

    data = np.vstack([
        np.random.multivariate_normal([1, 5], ([0.42323, 0], [0, 2]), 30),
        np.random.multivariate_normal([5, 2], ([0.2, 0], [0, 12]), 30),
        np.random.multivariate_normal([6, 4], ([0.7, 0], [0, 4]), 40)
    ])

    # 运行GMM Gibbs采样算法
    k = 3
    z, mu, pi, sigma = gmm_gibbs(data, k, 10, 0.3)

    # print("1:\n", sigma_1)
    # print("2:\n", sigma_2)
    # print("3:\n", sigma_3)
    # print("x:\n", x)
    print("均值:\n", mu)
    print("协方差:\n", sigma)
    print("混合系数:\n", pi)
    print("隐函数z:\n", z)
    