# myproject/utils/numerical/utils/derivative_utils_num.py

class DerivativeUtilsNum:
    def __init__(self, *args, g_func=None, f=None, x=None,
                 mu=None, i=None, j=None, h=1e-6):
       
        # obsługa pozycyjnych
        if len(args) >= 1 and g_func is None:
            g_func = args[0]
        if len(args) >= 2 and x is None:
            x = args[1]
        if len(args) >= 3 and mu is None:
            mu = args[2]
        if len(args) >= 4 and i is None:
            i = args[3]
        if len(args) >= 5 and j is None:
            j = args[4]
        if len(args) >= 6 and h == 1e-6:
            h = args[5]

        self.g_func = g_func
        self.f      = f
        self.x      = x
        self.mu     = mu
        self.i      = i
        self.j      = j
        self.h      = h

    def numerical_partial_g(self):
        # Teraz self.x zawsze jest np.ndarray
        x_ph = self.x.copy()
        x_mh = self.x.copy()
        x_ph[self.mu] += self.h
        x_mh[self.mu] -= self.h

        g_ph = self.g_func(x_ph)
        g_mh = self.g_func(x_mh)
        return (g_ph[self.i, self.j] - g_mh[self.i, self.j]) / (2 * self.h)

    def numerical_partial_scalar(self):
        x_ph = self.x.copy()
        x_mh = self.x.copy()
        x_ph[self.mu] += self.h
        x_mh[self.mu] -= self.h

        return (self.f(x_ph) - self.f(x_mh)) / (2 * self.h)
