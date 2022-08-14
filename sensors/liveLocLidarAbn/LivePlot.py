import random
import tkinter as Tk
from itertools import count

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np



index = count()

def animate(i):
    # Generate values
    x_vals.append(next(index))
    y_vals.append(np.tan(2*np.pi*random.random()))
    y_vals2.append(random.randint(0, 10))
    # Get all axes of figure
    ax1, ax2 = plt.gcf().get_axes()
    # Clear current data
    ax1.cla()
    ax2.cla()
    # Plot new data
    ax1.plot(x_vals, y_vals,linewidth=1)
    ax2.plot(x_vals, y_vals2,linewidth=1)



# GUI
mainWindow = Tk.Tk()

# graph 1
canvas = FigureCanvasTkAgg(plt.gcf(), master=mainWindow)

canvas.get_tk_widget().config(width=1500, height=750)
canvas.get_tk_widget().pack()
# Create two subplots in in two rows
plt.gcf().subplots(2, 1)
ani = FuncAnimation(plt.gcf(), animate, interval=10, blit=False)

# values for first graph
x_vals = []
y_vals = []
# values for second graph
y_vals2 = []

Tk.mainloop()

