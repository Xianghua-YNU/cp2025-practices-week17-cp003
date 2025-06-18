"""Module: Relaxation Method Solution
File: relaxation_method_solution.py
"""
import numpy as np
import matplotlib.pyplot as plt

def solve_ode(h, g, max_iter=10000, tol=1e-6):
    """
    使用松弛法求解抛体运动常微分方程
    d²x/dt² = -g ，边界条件为 x(0) = x(10) = 0
    
    参数:
        h (float): 时间步长
        g (float): 重力加速度
        max_iter (int): 最大迭代次数
        tol (float): 收敛容差
    返回:
        tuple: (时间数组, 解数组)
    """
    # 初始化时间数组
    t = np.arange(0, 10 + h, h)
    
    # 初始化解数组
    x = np.zeros(t.size)
    
    # 应用松弛迭代
    delta = 1.0
    iteration = 0
    
    while delta > tol and iteration < max_iter:
        x_new = np.copy(x)
        
        x_new[1:-1] = 0.5 * (h * h * g + x[2:] + x[:-2])
        
        # 计算最大变化
        delta = np.max(np.abs(x_new - x))
        
        # 更新解
        x = x_new
        iteration += 1
    
    return t, x

if __name__ == "__main__":
    # 问题参数
    h = 10.0 / 100  # 时间步长
    g = 9.8          # 重力加速度
    
    # 求解常微分方程
    t, x = solve_ode(h, g)
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(t, x, 'b-', linewidth=2, label='Projectile trajectory')
    plt.xlabel('time (s)')
    plt.ylabel('hight (m)')
    plt.title('Projectile Motion using Relaxation Method')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # 打印最大高度和时间
    max_height = np.max(x)
    max_time = t[np.argmax(x)]
    print(f"最大高度: {max_height:.2f} m，出现在 t = {max_time:.2f} s")
