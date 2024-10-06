#!/usr/bin/env python3
import numpy as np
import torch


class Precond:
    def __init__(self, name, theta1=0.2, theta2=2, delta=1 / 1.633):
        self.it = 0
        self.name = name
        self.theta1 = theta1
        self.theta2 = theta2
        self.delta = delta

    def get_alpha(self, s, m):
        alphas = np.linspace(0, 1, 1000)
        # line search of alpha

    def update_grad(self, cur_params, physics, grad, *args, **kwargs):
        if self.name == "static":
            grad = self._update_grad_static(cur_params, physics, grad, *args, **kwargs)
        elif self.name == "cheby":
            grad = self._update_grad_cheby(cur_params, physics, grad, *args, **kwargs)
        elif self.name == "dynamic":
            grad = self._update_grad_dynamic(cur_params, physics, grad, *args, **kwargs)
        return grad

    def _update_grad_static(self, cur_params, physics, grad_f, *args, **kwargs):
        """update the gradient with the static preconditioner"""

        alpha = cur_params["stepsize"]
        grad_f_preconditioned = physics.A_adjoint(physics.A(grad_f))

        grad_f_preconditioned *= -alpha
        grad_f_preconditioned += 2 * grad_f

        return grad_f_preconditioned

    def _update_grad_cheby(self, cur_params, physics, grad_f, *args, **kwargs):
        """update the gradient with the static cheby preconditioner"""

        alpha = cur_params["stepsize"]
        grad_f_preconditioned = physics.A_adjoint(physics.A(grad_f))
        grad_f_preconditioned *= -(10 / 3) * alpha
        grad_f_preconditioned += 4 * grad_f
        return grad_f_preconditioned

    def _update_grad_dynamic(self, cur_params, physics, grad_f, grad_f_prev, x, x_prev):
        """update the gradient with the dynamic preconditioner"""

        s = x - x_prev
        m = grad_f - grad_f_prev

        # precompute dot products
        sf = s.squeeze(0).squeeze(0).reshape(-1)
        mf = m.squeeze(0).squeeze(0).reshape(-1)

        ss = sf.dot(sf.conj()).real
        sm = sf.dot(mf.conj()).real
        mm = mf.dot(mf.conj()).real

        for a in np.linspace(0, 1, 1000):
            sv = a * ss + (1 - a) * sm
            vv = (a**2) * ss + ((1 - a) ** 2) * mm + (2 * a * (1 - a)) * sm
            if sv / ss >= self.theta1 and vv / sv <= self.theta2:
                break
        v = a * s + (1 - a) * m

        tau = ss / sv - torch.sqrt((ss / sv) ** 2 - ss / vv)

        tmp = sv - tau * vv
        grad_f_preconditioned = tau * grad_f

        if tmp >= self.delta * torch.sqrt(tmp * vv):
            u = sf - tau * vf
            u = u.dot(grad_f.squeeze(0).squeeze(0).reshape(-1)) * u
            u = u.reshape(grad_f.shape)

            grad_f_preconditioned += u / tmp
        return grad_f_preconditioned
