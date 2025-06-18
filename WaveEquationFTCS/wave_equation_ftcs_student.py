"""
学生模板：波动方程FTCS解
文件：wave_equation_ftcs_student.py
重要：函数名称必须与参考答案一致！
"""
"""
该模块使用FTCS（Forward-Time Central-Space）有限差分法求解一维波动方程。
波动方程形式：∂²u/∂t² = a² * ∂²u/∂x²
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def u_t(x, C=1, d=0.1, sigma=0.3, L=1):
    """
    计算初始速度分布ψ(x) = ∂u/∂t(t=0)
    Args:
        x (np.ndarray): 位置坐标数组
        C (float): 振幅常数，控制初始速度的强度
        d (float): 高斯分布中心偏移量
        sigma (float): 高斯分布宽度参数
        L (float): 弦的长度
    Returns:
        np.ndarray: 初始速度分布
    """
    # 公式：C * x(L-x)/L² * exp(-(x-d)²/(2σ²))
    # 该函数表示初始时刻(t=0)弦上各点的速度分布
    return C * x * (L - x) / L / L * np.exp(-(x - d)**2 / (2 * sigma**2))

def solve_wave_equation_ftcs(parameters):
    """
    使用FTCS有限差分法求解一维波动方程
    
    Args:
        parameters (dict): 包含模拟参数的字典:
            - 'a': 波速 (m/s)
            - 'L': 弦长度 (m)
            - 'd': 初始速度分布的偏移量 (m)
            - 'C': 初始速度分布的振幅常数 (m/s)
            - 'sigma': 初始速度分布的宽度参数 (m)
            - 'dx': 空间步长 (m)
            - 'dt': 时间步长 (s)
            - 'total_time': 总模拟时间 (s)
            
    Returns:
        tuple: 包含以下内容的元组:
            - u: 二维数组，波动方程的解 u(x, t)
            - x: 空间坐标数组
            - t: 时间坐标数组
    """
    # 从参数字典获取参数，使用默认值确保参数完整性
    a = parameters.get('a', 100)        # 波速 (默认100 m/s)
    L = parameters.get('L', 1)          # 弦长 (默认1 m)
    d = parameters.get('d', 0.1)        # 初始速度偏移 (默认0.1 m)
    C = parameters.get('C', 1)          # 初始速度振幅 (默认1 m/s)
    sigma = parameters.get('sigma', 0.3) # 初始速度分布宽度 (默认0.3 m)
    dx = parameters.get('dx', 0.01)     # 空间步长 (默认0.01 m)
    dt = parameters.get('dt', 5e-5)     # 时间步长 (默认0.00005 s)
    total_time = parameters.get('total_time', 0.1)  # 总时间 (默认0.1 s)

    # 创建空间和时间网格
    x = np.arange(0, L + dx, dx)  # 空间网格 (0到L，步长dx)
    t = np.arange(0, total_time + dt, dt)  # 时间网格 (0到total_time，步长dt)
    
    # 初始化解矩阵 u(x, t)
    u = np.zeros((x.size, t.size), float)

    # 计算稳定性参数 c = (a*dt/dx)²
    c_val = (a * dt / dx)**2
    
    # 检查CFL稳定性条件 (c < 1)
    if c_val >= 1:
        print(f"警告: 稳定性条件 c = {c_val} ≥ 1. 解可能不稳定.")

    # 设置初始条件:
    # 条件1: u(x, 0) = 0 (初始位移为零 - 弦静止)
    # 条件2: ∂u/∂t(t=0) = ψ(x) (初始速度分布)
    
    # 计算第一个时间步 (j=1) 的解:
    # 使用中心差分近似初始速度项
    # 公式: u_i,1 = c/2 * (u_i+1,0 + u_i-1,0) + (1-c) * u_i,0 + ψ(x_i) * dt
    # 由于初始位移 u_i,0 = 0, 简化为:
    # u_i,1 = ψ(x_i) * dt
    u[1:-1, 1] = u_t(x[1:-1], C, d, sigma, L) * dt

    # 使用FTCS格式进行后续时间步的迭代 (j ≥ 2)
    # 公式: u_i,j+1 = c * (u_i+1,j + u_i-1,j) + 2*(1-c)*u_i,j - u_i,j-1
    for j in range(1, t.size - 1):
        u[1:-1, j + 1] = c_val * (u[2:, j] + u[:-2, j]) + 2 * (1 - c_val) * u[1:-1, j] - u[1:-1, j - 1]

    return u, x, t

if __name__ == "__main__":
    # 演示和测试代码
    # 设置模拟参数
    params = {
        'a': 100,          # 波速 100 m/s
        'L': 1,             # 弦长 1 m
        'd': 0.1,           # 初始速度偏移 0.1 m
        'C': 1,             # 初始速度振幅 1 m/s
        'sigma': 0.3,       # 初始速度分布宽度 0.3 m
        'dx': 0.01,         # 空间步长 0.01 m
        'dt': 5e-5,         # 时间步长 0.00005 s
        'total_time': 0.1    # 总模拟时间 0.1 s
    }
    
    # 求解波动方程
    u_sol, x_sol, t_sol = solve_wave_equation_ftcs(params)

    # 创建动画可视化结果
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, xlim=(0, params['L']), ylim=(u_sol.min() * 1.1, u_sol.max() * 1.1))
    line, = ax.plot([], [], 'g-', lw=2)  # 创建绿色线条对象
    ax.set_title("一维波动方程求解 (FTCS方法)")
    ax.set_xlabel("位置 (m)")
    ax.set_ylabel("位移")

    # 动画更新函数
    def update(frame):
        """更新动画帧，显示特定时间步的波形"""
        line.set_data(x_sol, u_sol[:, frame])
        return line,

    # 创建动画对象
    # frames: 总帧数（时间步数）
    # interval: 帧间隔时间（毫秒）
    # blit: 使用blitting技术优化渲染
    ani = FuncAnimation(fig, update, frames=t_sol.size, interval=1, blit=True)
    
    # 显示动画
    plt.show()
    ani = FuncAnimation(fig, update, frames=t_sol.size, interval=1, blit=True)
    plt.show()
