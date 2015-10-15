import tkinter as tk
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import heatEquation


class LabelledIntegralInputFrame:
    ''' Contains : a label and an input on its right'''

    def __init__(self, parent, txt, value):
        ''' 'parent' is the parent frame
        'txt' is the label text
        'value' is the tk.IntVar input value
        '''

        self.frame = tk.Frame(parent)

        self.label = tk.Label(self.frame, text=txt)
        self.input = tk.Entry(self.frame, textvariable=value, width=5)

        self.label.grid(row=0, column=0, sticky=tk.W)
        self.input.grid(row=0, column=1, sticky=tk.W)



class ParameterFrame:
    ''' Contains :
    - 1 LabelledIntegralInputFrame to change the size of the plane studied
    - 1 Button to apply the change
    '''

    def __init__(self, main):
        '''
        'main' is the MainFrame in which this ParameterFrame is
        embedded, which has the attribute 'size'
        '''

        self.frame = tk.LabelFrame(main.frame, text='Parameters', padx=20, pady=20)
        
        # Resizing Configuration
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)

        # Widgets
        self.getSize = LabelledIntegralInputFrame(self.frame, "Size : ", main.size)
        self.apply = tk.Button(self.frame, text="Apply", command=main.resetInput)

        self.getSize.frame.grid(row=0, column=0, sticky=tk.W)
        self.apply.grid(row=0, column=1, sticky=tk.W)
        



class InputFrame:
    '''
    Contains a Canvas where a grid is drawn  
    '''

    def __init__(self, main):
        '''
        'main' is the MainFrame in which this ParameterFrame is
        embedded, which has the attribute input to modify
        '''

        self.main = main
        self.frame = tk.LabelFrame(main.frame, text='Heat flux density', 
                                   padx=20, pady=20)

        self.colormap = ['#000084', '#0000ff', '#006dff', '#00e1fb', '#00e1fb',
                         '#beff39', '#ffd000', '#ff6400', '#da0000', '#800000']

        # Resizing configuration
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        # Initializing the canvas
        self.canvas = tk.Canvas(self.frame, bg='white')
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<Button-1>", self.clickOnCell)
        self.drawContent()



    def drawContent(self):
        '''Reset the input heat flux density with current dimensions
        then draw content'''

        size = self.main.size.get()
        canvasSize = 10 * size

        self.main.input = [[0 for x in range(size)] for y in range(size)]

        # updates canvas dimensions
        self.canvas.config(width=canvasSize, height=canvasSize)
        self.canvas.delete('all')

        # draws grid
        for i in range(size):
            self.canvas.create_line(0, 10*i, canvasSize, 10*i)        
            self.canvas.create_line(10*i, 0, 10*i, canvasSize)
        
        # fills grid
        self.rectangles = []

        for row in range(size):
            self.rectangles.append([])
            y0 = 10 * row + 1
            y1 = 10 * (row+1) -1
            for col in range(size):
                x0 = 10 * col + 1
                x1 = 10 * (col+1) - 1
                c = self.main.input[row][col]
                color = self.colormap[c]
                nextColor = self.colormap[(c+1)%len(self.colormap)]
                rect = self.canvas.create_rectangle(x0, y0, x1, y1, 
                                                    fill=color,
                                                    activefill=nextColor)
                self.rectangles[row].append(rect)


    def clickOnCell(self, event):
        '''
        Function triggered by a click on the input grid
        Increment the value of corresponding cell and changes its color
        '''

        row = event.y//10 
        col = event.x//10

        # change color to next one
        c = self.main.input[row][col]
        self.main.input[row][col] = (c+1)%len(self.colormap)
        c = self.main.input[row][col]

        # modify rectangle
        color = self.colormap[c]
        nextColor = self.colormap[(c+1)%len(self.colormap)]
        self.canvas.itemconfig(self.rectangles[row][col], fill=color,
                               activefill=nextColor)





class SolutionFrame:
    ''' 
    Contains a canvas where is embedded a pyplot obtained
    by calling heatEquation.solveHeatEquation on the input
    '''

    def __init__(self, main):
        '''
        'main' is the MainFrame in which this ParameterFrame is
        embedded, which has the attribute input to modify
        '''
        self.main = main
        self.frame = tk.LabelFrame(main.frame, text='Temperature map', 
                                   padx=20, pady=20)

        # Resizing configuration
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        # displaying solution
        self.updatePlot(main)
        
    

    def updatePlot(self, main):
        size, h, airConductivity = 50, 0.01, 0.025

        # Figure initialization
        self.fig = plt.figure(dpi=100)
        self.plot = self.fig.add_subplot(111)
        self.plot.axis('off')
        self.sol = heatEquation.solveHeatEquation(np.array(main.input), h, airConductivity).transpose()
        self.plot.imshow(self.sol, interpolation='bilinear', cmap=cm.jet_r)

        # Canvas initialization
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.canvas.draw()
        

    

class MainFrame:
    ''' This class represents the main frame'''

    def __init__(self, frame):
        # Resizing configuration
        self.frame = frame
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=0)
        self.frame.columnconfigure(2, weight=1)
        self.frame.rowconfigure(0, weight=0)
        self.frame.rowconfigure(1, weight=1)

        # global variables :
        self.size = tk.IntVar()
        self.size.set(20)
        self.input = [[0 for x in range(self.size.get())] 
                       for y in range(self.size.get())]

        # Widgets
        self.paramFrame = ParameterFrame(self)
        self.inputFrame = InputFrame(self)
        self.solutionFrame = SolutionFrame(self)
        self.solve = tk.Button(self.frame, text="-> Solve ->", 
                               command=self.updateSolution)
        
        self.paramFrame.frame.grid(row=0, column=0, columnspan=3, 
                                   sticky=tk.N+tk.S+tk.E+tk.W)
        self.inputFrame.frame.grid(row=1, column=0, 
                                   sticky=tk.N+tk.S+tk.E+tk.W)
        self.solutionFrame.frame.grid(row=1, column=2, 
                                      sticky=tk.N+tk.S+tk.E+tk.W)
        self.solve.grid(row=1, column=1)


    def resetInput(self):
        self.inputFrame.drawContent()

    def updateSolution(self):
        self.solutionFrame.updatePlot(self)


# ROOT
root = tk.Tk()
# allows quitting by clicking the 'X'       
root.protocol("WM_DELETE_WINDOW", quit)
# Resizing configuration
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# MAINFRAME
main = tk.Frame(root, padx=20, pady=20)
main.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
main = MainFrame(main) 


root.mainloop()