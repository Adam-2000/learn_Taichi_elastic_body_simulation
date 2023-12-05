# -*- coding: utf-8 -*-
# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
# https://www.bilibili.com/video/BV1ZK411H7Hc?p=4
# https://github.com/ShaneFX/GAMES201/tree/master/HW01
# https://www.bilibili.com/video/BV1934y1X7MD?p=6&vd_source=a2e8a29ea5b336ff1a71c971ab786a47

import numpy as np

import taichi as ti

res = 512
dt = 0.03
p_jacobi_iters = 500  # 40 for a quicker but less accurate result
f_strength = 10000.0
time_c = 2
maxfps = 60
dye_decay = 1 - 1 / (maxfps * time_c)
force_radius = res / 2.0

ti.init(arch=ti.cuda)

viscosity_niu = ti.field(float, shape=())

_velocities = ti.Vector.field(2, float, shape=(res, res))
_new_velocities = ti.Vector.field(2, float, shape=(res, res))
velocity_divs = ti.field(float, shape=(res, res))
velocity_laplaces = ti.Vector.field(2, float, shape=(res, res))
velocity_curls = ti.field(float, shape=(res, res))
_pressures = ti.field(float, shape=(res, res))
_new_pressures = ti.field(float, shape=(res, res))
_dye_buffer = ti.Vector.field(3, float, shape=(res, res))
_new_dye_buffer = ti.Vector.field(3, float, shape=(res, res))


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)

@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = ti.max(0, ti.min(res - 1, I))
    return qf[I]


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


# 3rd order Runge-Kutta
@ti.func
def backtrace(vf: ti.template(), p, dt_: ti.template()):
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt_ * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt_ * v2
    v3 = bilerp(vf, p2)
    p -= dt_ * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p


@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        p = backtrace(vf, p, dt)
        new_qf[i, j] = bilerp(qf, p) * dye_decay

@ti.kernel
def advect_vel(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        p = backtrace(vf, p, dt)
        new_qf[i, j] = bilerp(qf, p)

@ti.kernel
def apply_impulse(vf: ti.template(), dyef: ti.template(), imp_data: ti.types.ndarray()):
    g_dir = -ti.Vector([0, 9.8]) * 300
    for i, j in vf:
        omx, omy = imp_data[2], imp_data[3]
        mdir = ti.Vector([imp_data[0], imp_data[1]])
        dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
        d2 = dx * dx + dy * dy
        factor = ti.exp(-d2 / force_radius)

        dc = dyef[i, j]
        # a = dc.norm()

        # momentum = (mdir * f_strength * factor + g_dir * a / (1 + a)) * dt
        momentum = (mdir * f_strength * factor + g_dir + viscosity_niu[None] * velocity_laplaces[i, j]) * dt
        # momentum = (mdir * f_strength * factor + g_dir) * dt

        vf[i, j] += momentum
        # add dye
        if mdir.norm() > 0.5:
            dc += ti.exp(-d2 * (4 / (res / 15) ** 2)) * ti.Vector([imp_data[4], imp_data[5], imp_data[6]])

        dyef[i, j] = dc


@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        vc = sample(vf, i, j)
        if i == 0:
            vl.x = -vc.x
        if i == res - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == res - 1:
            vt.y = -vc.y
        velocity_divs[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5

@ti.kernel
def laplace(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        vc = sample(vf, i, j)
        if i == 0:
            vl.x = -vc.x
        if i == res - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == res - 1:
            vt.y = -vc.y
        velocity_laplaces[i, j] = 4 * vc - vr - vl - vt - vb
        
@ti.kernel
def vorticity(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        velocity_curls[i, j] = (vr.y - vl.y - vt.x + vb.x) * 0.5


@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])


def solve_pressure_jacobi():
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()


def step(mouse_data):
    advect_vel(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)
    velocities_pair.swap()
    dyes_pair.swap()

    laplace(velocities_pair.cur)
    
    apply_impulse(velocities_pair.cur, dyes_pair.cur, mouse_data)

    divergence(velocities_pair.cur)

    solve_pressure_jacobi()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)


class MouseDataGen:
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None

    def __call__(self, gui):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        # [4:7]: color
        mouse_data = np.zeros(8, dtype=np.float32)
        if gui.is_pressed(ti.GUI.LMB):
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) * res
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                # Set lower bound to 0.3 to prevent too dark colors
                self.prev_color = (np.random.rand(3) * 0.7) + 0.3
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data


def reset():
    velocities_pair.cur.fill(0)
    pressures_pair.cur.fill(0)
    dyes_pair.cur.fill(0)


def main():
    global debug, curl_strength
    visualize_d = True  # visualize dye (default)
    visualize_v = False  # visualize velocity
    visualize_c = False  # visualize curl
    visualize_l = False  # visualize laplace

    paused = False

    gui = ti.GUI("Stable Fluid", (res, res))
    md_gen = MouseDataGen()

    viscosity_niu[None] = .1
    frame = 0
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == "r":
                paused = False
                reset()
            elif e.key == "s":
                if curl_strength:
                    curl_strength = 0
                else:
                    curl_strength = 7
            elif e.key == ti.GUI.UP:
                viscosity_niu[None] *= 2
            elif e.key == ti.GUI.DOWN:
                viscosity_niu[None] /= 2
            elif e.key == "v":
                visualize_v = True
                visualize_c = False
                visualize_d = False
                visualize_l = False
            elif e.key == "d":
                visualize_d = True
                visualize_v = False
                visualize_c = False
                visualize_l = False
            elif e.key == "c":
                visualize_c = True
                visualize_d = False
                visualize_v = False
                visualize_l = False
            elif e.key == "l":
                visualize_l = True
                visualize_d = False
                visualize_v = False
                visualize_c = False
            elif e.key == "p":
                paused = not paused
            elif e.key == "d":
                debug = not debug

        # Debug divergence:
        # print(max((abs(velocity_divs.to_numpy().reshape(-1)))))

        if not paused:
            mouse_data = md_gen(gui)
            step(mouse_data)
        if visualize_c:
            vorticity(velocities_pair.cur)
            gui.set_image(velocity_curls.to_numpy() * 0.03 + 0.5)
        elif visualize_d:
            gui.set_image(dyes_pair.cur)
        elif visualize_v:
            gui.set_image(velocities_pair.cur.to_numpy() * 0.01 + 0.5)
        elif visualize_l:
            gui.set_image(velocity_laplaces.to_numpy() * 1 + 0.5)
        
        gui.text(
            content=f"Y: Viscosity: {viscosity_niu[None]:.3f}",
            pos=(0, 0.99),
            color=0xFFFFFFFF,
        )
        # gui.show(f"frames/frame_{frame:05d}.png")
        frame += 1
        gui.show()
        
if __name__ == "__main__":
    main()