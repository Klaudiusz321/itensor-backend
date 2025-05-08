class DerivativeUtilsNum:
    def __init__(self, g_func=None, f=None, x=None,
                 mu=None, i=None, j=None, h=1e-6):
        self.g_func = g_func
        self.f      = f           # do scalar
        self.x      = x
        self.mu     = mu
        self.i      = i
        self.j      = j
        self.h      = h

    def numerical_partial_g(self):
        x_ph, x_mh = self.x.copy(), self.x.copy()
        x_ph[self.mu] += self.h; x_mh[self.mu] -= self.h
        g_ph, g_mh = self.g_func(x_ph), self.g_func(x_mh)
        return (g_ph[self.i, self.j] - g_mh[self.i, self.j])/(2*self.h)

    def numerical_partial_scalar(self):
        x_ph, x_mh = self.x.copy(), self.x.copy()
        x_ph[self.mu] += self.h; x_mh[self.mu] -= self.h
        return (self.f(x_ph) - self.f(x_mh))/(2*self.h)
