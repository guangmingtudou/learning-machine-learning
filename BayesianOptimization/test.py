import housepricepredict
import numpy as np
import matplotlib.pyplot as plt
import os

batch_size = np.arange(5, 60)
lr = 0.01
np.random.seed(0)

result = []
# for bs in batch_size:
#     epochs, loss = housepricepredict.housepricepredict(bs, lr)
#     result.append(5e-4 *epochs + loss)

def plot_line_chart(x, y, title='连线图', xlabel='X轴', ylabel='Y轴', color='blue', 
                    linewidth=2, linestyle='-', marker='o', marker_size=8, 
                    grid=True, figsize=(10, 6), save_path=None, show=True):
    """
    绘制连线图
    
    参数:
    x: X轴数据
    y: Y轴数据
    title: 图表标题
    xlabel: X轴标签
    ylabel: Y轴标签
    color: 线条颜色
    linewidth: 线条宽度
    linestyle: 线条样式 ('-', '--', '-.', ':')
    marker: 数据点标记 ('o', 's', '^', 'D', '*', '.', 'x', '+')
    marker_size: 标记大小
    grid: 是否显示网格
    figsize: 图表尺寸 (宽, 高)
    save_path: 保存路径（如 'chart.png'），None则不保存
    show: 是否显示图表
    """
    # 创建图形和坐标轴
    plt.figure(figsize=figsize)
    
    # 绘制连线图
    plt.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle,
             marker=marker, markersize=marker_size, markerfacecolor='white',
             markeredgecolor=color, markeredgewidth=1.5)
    
    # 设置标题和标签
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # 显示网格
    if grid:
        plt.grid(True, linestyle='--', alpha=0.6)
    
    # 自动调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    # 显示图表
    if show:
        plt.show()
    
    return plt.gcf()  # 返回图形对象，便于进一步操作



batch_size = 16
lr = np.linspace(0.001, 0.1, 100)
np.random.seed(0)

result = []
for l in lr:
    epochs, loss = housepricepredict.housepricepredict(batch_size, l)
    result.append(1e-4 *epochs + loss)

plot_line_chart(lr, result)