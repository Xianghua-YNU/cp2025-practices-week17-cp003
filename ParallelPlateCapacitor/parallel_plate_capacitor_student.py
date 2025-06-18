"""学生模板：ParallelPlateCapacitor
文件：parallel_plate_capacitor_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def solve_laplace_jacobi(xgrid, ygrid, w, d, tol=1e-5):
    """
    使用Jacobi迭代法求解拉普拉斯方程
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        tol (float): 收敛容差
    
    返回:
        tuple: (potential_array, iterations, convergence_history)
    
    物理背景: 求解平行板电容器内部的电势分布，满足拉普拉斯方程 \(\nabla^2 V = 0\)。
    数值方法: 使用Jacobi迭代法，通过反复迭代更新每个网格点的电势值，直至收敛。
    
    实现步骤:
    1. 初始化电势网格，设置边界条件（极板电势）。
    2. 循环迭代，每次迭代根据周围点的电势更新当前点的电势。
    3. 记录每次迭代的最大变化量，用于收敛历史分析。
    4. 检查收敛条件，如果最大变化量小于容差，则停止迭代。
    5. 返回最终的电势分布、迭代次数和收敛历史。
    """
    # 初始化电势网格（全零矩阵）
    u = np.zeros((ygrid, xgrid))
    
    # 计算平行板位置（居中放置）
    xL = (xgrid - w) // 2    # 左边界
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # 设置边界条件（极板电势）
    u[yT, xL:xR+1] = 100.0  # Top plate: +100V    # 上极板: +100V
    u[yB, xL:xR+1] = -100.0  # Bottom plate: -100V
    
    iterations = 0    # 迭代计数器
    max_iter = 10000    # 最大迭代次数
    convergence_history = []    # 收敛历史记录
    
    while iterations < max_iter:
        u_old = u.copy()
        
        # Jacobi迭代核心公式：内部点使用四邻域平均值更新
        # 注意：边界点保持固定不变
        u[1:-1,1:-1] = 0.25*(u[2:,1:-1] + u[:-2,1:-1] + u[1:-1, 2:] + u[1:-1,:-2]) 

        # 保持边界条件（每次迭代后重置极板电势）
        u[yT, xL:xR+1] = 100.0
        u[yB, xL:xR+1] = -100.0
        
        # 计算收敛指标：当前迭代的最大变化量
        max_change = np.max(np.abs(u - u_old))
        convergence_history.append(max_change)

        # 检查收敛条件
        iterations += 1
        if max_change < tol:
            break
    
    return u, iterations, convergence_history

def solve_laplace_sor(xgrid, ygrid, w, d, omega=1.25, Niter=1000, tol=1e-5):
    """
    实现SOR算法求解平行板电容器的电势分布
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        omega (float): 松弛因子
        Niter (int): 最大迭代次数
        tol (float): 收敛容差
    返回:
        tuple: (电势分布数组, 迭代次数, 收敛历史)
    
    物理背景: 求解平行板电容器内部的电势分布，满足拉普拉斯方程 \(\nabla^2 V = 0\)。
    数值方法: 使用逐次超松弛（SOR）迭代法，通过引入松弛因子加速收敛。
    
    实现步骤:
    1. 初始化电势网格，设置边界条件（极板电势）。
    2. 循环迭代，每次迭代根据周围点和松弛因子更新当前点的电势。
    3. 记录每次迭代的最大变化量，用于收敛历史分析。
    4. 检查收敛条件，如果最大变化量小于容差，则停止迭代。
    5. 返回最终的电势分布、迭代次数和收敛历史。
    """
    # 初始化电势网格
    u = np.zeros((ygrid, xgrid))
    
    # 计算平行板位置
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # 设置边界条件
    u[yT, xL:xR+1] = 100.0  # 上极板
    u[yB, xL:xR+1] = -100.0  
    
    convergence_history = []    # 收敛历史

    # SOR迭代主循环
    for iteration in range(Niter):
        u_old = u.copy()    # 保存当前状态
        
        # 遍历所有内部点（行优先）
        for i in range(1, ygrid-1):    # 行索引
            for j in range(1, xgrid-1):
                # 跳过极板区域（保持边界条件不变）
                if (i == yT and xL <= j <= xR) or (i == yB and xL <= j <= xR):
                    continue
                
                # 计算残差（拉普拉斯离散格式）
                r_ij = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
                
                # SOR公式：结合当前值和残差进行更新
                u[i, j] = (1 - omega) * u[i, j] + omega * r_ij
        
        # 保持边界条件（重置极板电势）
        u[yT, xL:xR+1] = 100.0
        u[yB, xL:xR+1] = -100.0
        
        # 计算收敛指标
        max_change = np.max(np.abs(u - u_old))
        convergence_history.append(max_change)
        
        # 检查收敛条件
        if max_change < tol:
            break
    
    return u, iteration + 1, convergence_history
    
def plot_results(x, y, u, method_name):
    """
    绘制三维电势分布、等势线和电场线
    
    参数:
        x (array): X坐标数组
        y (array): Y坐标数组
        u (array): 电势分布数组
        method_name (str): 方法名称
    
    实现步骤:
    1. 创建包含两个子图的图形。
    2. 在第一个子图中绘制三维线框图显示电势分布以及在z方向的投影等势线。
    3. 在第二个子图中绘制等势线和电场线流线图。
    4. 设置图表标题、标签和显示(注意不要出现乱码)。
    """
    fig = plt.figure(figsize=(10, 5))
    
    # 创建3D线框子图
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(x, y)    # 创建网格坐标
    # 绘制3D线框电势图
    ax1.plot_wireframe(X, Y, u, alpha=0.7)
    # 在z=min平面添加等势线投影
    levels =np.linspace(u.min(),u.max(),20)
    ax1.contour(x, y, u, zdir = 'z', offset = u.min(),levels = levels)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Potential (V)')
    ax1.set_title(f'3D Potential Distribution\n({method_name})')
    
    # 创建二维等势线和电场线子图
    ax2 = fig.add_subplot(122)
    levels = np.linspace(u.min(), u.max(), 20)
    # 绘制等势线（红色虚线）
    contour = ax2.contour(X, Y, u, levels=levels, colors='red', linestyles='dashed', linewidths=0.8)
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

    # 计算电场强度（电势的负梯度）
    EY, EX = np.gradient(-u, 1) # 注意：np.gradient返回(行方向, 列方向) = (y, x)
    # 绘制电场线流线图（蓝色箭头）
    ax2.streamplot(X, Y, EX, EY, density=1.5, color='blue', linewidth=1, arrowsize=1.5, arrowstyle='->')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Equipotential Lines & Electric Field Lines\n({method_name})')
    ax2.set_aspect('equal')    # 保持纵横比一致
    
    plt.tight_layout()    # 自动调整子图间距
    plt.show()

if __name__ == "__main__":
    # 设置模拟参数
    xgrid, ygrid = 50, 50    # 网格尺寸
    w, d = 20, 20  # 极板宽度和间距
    tol = 1e-3    # 收敛容差
    # 创建坐标数组（物理空间）
    x = np.linspace(0, xgrid-1, xgrid)
    y = np.linspace(0, ygrid-1, ygrid)
    
    print("Solving Laplace equation for parallel plate capacitor...")
    print(f"Grid size: {xgrid} x {ygrid}")
    print(f"Plate width: {w}, separation: {d}")
    print(f"Tolerance: {tol}")
    
    # 使用Jacobi方法求解
    print("1. Jacobi iteration method:")
    start_time = time.time()
    u_jacobi, iter_jacobi, conv_history_jacobi = solve_laplace_jacobi(xgrid, ygrid, w, d, tol=tol)
    time_jacobi = time.time() - start_time
    print(f"   Converged in {iter_jacobi} iterations")
    print(f"   Time: {time_jacobi:.3f} seconds")
    
    # 使用SOR方法求解
    print("2. Gauss-Seidel SOR iteration method:")
    start_time = time.time()
    u_sor, iter_sor, conv_history_sor = solve_laplace_sor(xgrid, ygrid, w, d, tol=tol)
    time_sor = time.time() - start_time
    print(f"   Converged in {iter_sor} iterations")
    print(f"   Time: {time_sor:.3f} seconds")
    
    # 性能比较
    print("\n3. Performance comparison:")
    print(f"   Jacobi: {iter_jacobi} iterations, {time_jacobi:.3f}s")
    print(f"   SOR:    {iter_sor} iterations, {time_sor:.3f}s")
    print(f"   Speedup: {iter_jacobi/iter_sor:.1f}x iterations, {time_jacobi/time_sor:.2f}x time")
    
    # 可视化结果
    plot_results(x, y, u_jacobi, "Jacobi Method")
    plot_results(x, y, u_sor, "SOR Method")
    
    # 绘制收敛历史比较图
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(conv_history_jacobi)), conv_history_jacobi, 'r-', label='Jacobi Method')
    plt.semilogy(range(len(conv_history_sor)), conv_history_sor, 'b-', label='SOR Method')
    plt.xlabel('Iteration')
    plt.ylabel('Maximum Change')
    plt.title('Convergence Comparison')
    plt.grid(True)
    plt.legend()
    plt.show()
