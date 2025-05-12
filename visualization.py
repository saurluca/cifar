import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable

@torch.no_grad()
def create_contourline_figure(fn : Callable, log_scale : bool = False ) -> plt.Figure:
    x = torch.linspace(-4, 4, steps=100)
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    z = fn(xy) if not log_scale else fn(xy).log()

    fig, ax = plt.subplots(1,1)
    ax.contour(
        xy[:,0].reshape(100, 100), 
        xy[:,1].reshape(100, 100), 
        z.reshape(100, 100),
        levels = 50
    )
    ax.scatter(1,1, label = "Minimum", c="orange", zorder=20)
    return fig

@torch.no_grad()
def create_trapezoid_figure(q1: tuple[int], q2: tuple[int], q3: tuple[int], q4: tuple[int]) -> plt.Figure:
    fig, ax = plt.subplots(1,1)
    # Plot the trapezoid
    ax.plot([q1[0], q2[0]], [q1[1], q2[1]], color='b')
    ax.plot([q2[0], q3[0]], [q2[1], q3[1]], color='b')
    ax.plot([q3[0], q4[0]], [q3[1], q4[1]], color='b')
    ax.plot([q4[0], q1[0]], [q4[1], q1[1]], color='b')
    
    # Set plot limits
    ax.set_xlim(min(q1[0], q2[0], q3[0], q4[0]) - 1, max(q1[0], q2[0], q3[0], q4[0]) + 1)
    ax.set_ylim(min(q1[1], q2[1], q3[1], q4[1]) - 1, max(q1[1], q2[1], q3[1], q4[1]) + 1)
    
    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trapezoid Visualization')
    
    # Show plot
    ax.grid(True)
    #plt.gca().set_aspect('equal', adjustable='box')
    return fig

@torch.no_grad()
def scatter_points_and_labels(fig: plt.Figure, points: torch.Tensor, labels: torch.Tensor, name: str):
    ax = fig.axes[0]
    true_points = points[labels == 0]
    x, y = true_points.split(1, dim=-1)
    ax.scatter(x, y, label = f"{name} - label 0", color='orange', zorder=19)
    false_points = points[labels == 1]
    x, y = false_points.split(1, dim=-1)
    ax.scatter(x, y, label = f"{name} - label 1", color='cyan', zorder=19)
    return fig

@torch.no_grad()
def scatter_path_in_figure(fig : plt.Figure, path : torch.Tensor, name : str) -> plt.Figure:
    x, y = path.split(1, dim=-1)
    ax = fig.axes[0]
    ax.set_title("Loss curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.scatter(x, y, label = name, zorder=19)
    return fig

@torch.no_grad()
def create_losscurve_figure(loss_values : list[float]) -> plt.Figure:
    fig, ax = plt.subplots(1,1)
    ax.plot(loss_values)
    return fig   

def show_figure(fig : plt.Figure) -> None:
    fig.legend()
    fig.show()