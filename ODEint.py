import numpy as np

class RungeKutta4():
    def __init__(self, f, step = 1, stopping_criterion = lambda state: True, verbose = False):

        self.f = f
        self.step = step
        self.verbose = verbose
        self.stopping_criterion = stopping_criterion
        
        self.a = np.array([
            [0, 0, 0, 0],
            [1/2, 0, 0, 0],
            [0, 1/2, 0, 0],
            [0, 0, 1, 0],
        ])

        self.b = np.array(
            [1/6, 1/3, 1/3, 1/6]
            )

        self.c = np.array([
            0, 1/2, 1/2, 1
        ])

    def integrate(self, x_start, x_end, y_start):

        x = [x_start]
        y = [y_start]

        h = self.step

        try:
            while abs(x[-1]) <= abs(x_end):
                
                if self.stopping_criterion(y[-1]):
                        
                    next = self.next_step(x[-1], y[-1], self.step)
                    
                    x.append(next[0])
                    y.append(next[1])
                        
                else:
                    break
    
        except KeyboardInterrupt:
            if self.verbose:
                print("Integration stopped.")
        
        return x, y


    def next_step(self, x, y, h):
        k = np.zeros((4, len(y)))
            
        for i in range(4):
            k[i,:] = h*self.f(y + np.dot(self.a[i], k))
        
        y1 = y + np.dot(self.b, k)
        
        x1 = x + h
        
        return x1, y1

class RungeKuttaFehlberg45():
    def __init__(self, f, initial_step = 1, AccuracyGoal = 10, PrecisionGoal = 0, SF = 0.84, verbose = False, hmax = 1e+16):

        self.f = f
        self.tolerance = 1
        self.Atol = 10**(-AccuracyGoal)
        self.Rtol = 10**(-PrecisionGoal)
        self.initial_step = initial_step
        self.verbose = verbose
        self.hmax = hmax
        self.SF = SF

        self.a = np.array([
            [0, 0, 0, 0, 0, 0],
            [1/4, 0, 0, 0, 0, 0],
            [3/32, 9/32, 0, 0, 0, 0],
            [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
            [439/216, -8, 3680/513, -845/4104, 0, 0],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0],
        ])

        self.b = np.array([
            [16/135,  0, 6656/12825, 28561/56430, -9/50, 2/55],
            [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
        ])

        self.c = np.array([
            0, 1/4, 3/8, 12/13, 1, 1/2
        ])

    def integrate(self, x_start, x_end, y_start, initial_step, args = []):

        x = [x_start]
        y = [y_start]

        h = initial_step

        try:
            while abs(x[-1]) <= abs(x_end):
                    
                next = self.next_step(x[-1], y[-1], h, args)
                x.append(next[0])
                y.append(next[1])
                h = next[2]
    
        except KeyboardInterrupt:
            if self.verbose:
                print("Integration stopped.")
            exit = "stopped"
        
        return x, y


    def next_step(self, x, y, h, args):
        k = np.zeros((6, len(y)))
        h1 = h
        while True:
            
            for i in range(6):
                k[i,:] = h1*self.f(*(y + np.dot(self.a[i], k)), args)
            
            y4 = y + np.dot(self.b[1], k)
            y5 = y + np.dot(self.b[0], k)
            
            v = y4-y5

            err = np.linalg.norm(v)/h1/self.Atol

            if err > 1:
                delta = self.SF*err**(-0.2)
                if delta <= 0.1:
                    h1 *= 0.1
                elif delta >= 4:
                    h1 *= 4
                else:
                    h1 *= delta
                continue
            else:
                if err == 0:
                    if self.hmax > 0:
                        h2 = min(self.hmax, 2*h1)
                    else:
                        h2 = max(self.hmax, 2*h1)
                    break
                if self.hmax > 0:
                    h2 = min(self.hmax, h1*self.SF*err**(-0.2))
                else:
                    h2 = max(self.hmax, h1*self.SF*err**(-0.2))
                break
            
        x1 = x + h1
        return x1, y4, h2