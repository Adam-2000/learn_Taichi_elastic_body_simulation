# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:30:04 2023

@author: 45242
"""

import taichi as ti
import taichi.math as tm
import sys

# ti.init(arch=ti.cpu, debug=True)
ti.init(arch=ti.cpu)
paused = ti.field(dtype=ti.i32, shape=())
drag_damping = ti.field(dtype=ti.f32, shape=())
dashpot_damping = ti.field(dtype=ti.f32, shape=())
miu_S = ti.field(dtype=ti.f32, shape=())
lambda_S = ti.field(dtype=ti.f32, shape=())

max_num_particles = 1024
max_num_triangles = 1024
dt = 1e-3
substeps = 10

num_particles = ti.field(dtype=ti.i32, shape=())
x = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
xk = ti.Vector.field(2, dtype=ti.f32, shape=(4, max_num_particles))
vk = ti.Vector.field(2, dtype=ti.f32, shape=(4, max_num_particles))
fk = ti.Vector.field(2, dtype=ti.f32, shape=(4, max_num_particles))
m = ti.field(dtype=ti.f32, shape=max_num_particles)
f = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
fixed = ti.field(dtype=ti.i32, shape=max_num_particles)

num_triangles = ti.field(dtype=ti.i32, shape=())
triangles = ti.field(dtype=ti.i32, shape=(max_num_triangles, 3))
volumns = ti.field(dtype=ti.f32, shape=max_num_triangles)
inverse_Xs = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_num_triangles)
area_vectors = ti.Vector.field(2, dtype=ti.f32, shape=(max_num_triangles,3))
F = ti.Matrix.field(2, 2, dtype=ti.f32, shape = max_num_triangles)
M_buffer = ti.Matrix.field(2, 2, dtype=ti.f32, shape = max_num_triangles)
connected = ti.field(dtype=ti.i32, shape=(max_num_particles, max_num_particles))

@ti.func 
def build_triangle(a, b, c):
    v1 = ti.Vector([x[b][0]-x[a][0], x[b][1]-x[a][1], 0])
    v2 = ti.Vector([x[c][0]-x[a][0], x[c][1]-x[a][1], 0])
    idx = ti.atomic_add(num_triangles[None], 1)
    v_cross = tm.cross(v1, v2)
    volumns[idx] = 0.5 * v_cross.norm()
    if v_cross[2] >= 0:
       triangles[idx, 0] = a
       triangles[idx, 1] = b
       triangles[idx, 2] = c
    else:
       triangles[idx, 0] = a
       triangles[idx, 1] = c
       triangles[idx, 2] = b
    X = tm.mat3(1.0)
    X[0, 0], X[1, 0] = x[triangles[idx, 0]][0], x[triangles[idx, 0]][1]
    X[0, 1], X[1, 1] = x[triangles[idx, 1]][0], x[triangles[idx, 1]][1]
    X[0, 2], X[1, 2] = x[triangles[idx, 2]][0], x[triangles[idx, 2]][1]
    inverse_Xs[idx] = tm.inverse(X)
    for i in ti.static(range(3)):
        m[triangles[idx, i]] += volumns[idx] / 3
        v_temp = x[triangles[idx, (i + 2) % 3]] - x[triangles[idx, (i + 1) % 3]]
        area_vectors[idx, i] = [v_temp[1], -v_temp[0]]
    
@ti.func 
def add_line(a, b):
    connected[a, b] = 1
    connected[b, a] = 1
@ti.kernel 
def generate_obj():
    x[0] = [0.3, 0.3]
    v[0] = [0, 0]
    f[0] = [0, 0]
    m[0] = 0
    x[1] = [0.3, 0.4]
    v[1] = [0, 0]
    f[1] = [0, 0]
    m[1] = 0
    x[2] = [0.4, 0.4]
    v[2] = [0, 0]
    f[2] = [0, 0]
    m[2] = 0
    build_triangle(0, 1, 2)
    add_line(0, 1)
    add_line(0, 2)
    add_line(2, 1)
    num_particles[None] = 3
       
@ti.func 
def substep1(iter_i):
    for c in range(num_triangles[None]):
        X = tm.mat3(1.0)
        X[0, 0], X[1, 0] = xk[iter_i, triangles[c, 0]][0], xk[iter_i, triangles[c, 0]][1]
        X[0, 1], X[1, 1] = xk[iter_i, triangles[c, 1]][0], xk[iter_i, triangles[c, 1]][1]
        X[0, 2], X[1, 2] = xk[iter_i, triangles[c, 2]][0], xk[iter_i, triangles[c, 2]][1]
        X = X @ inverse_Xs[c]
        F[c][0, 0], F[c][0, 1] = X[0, 0], X[0, 1]
        F[c][1, 0], F[c][1, 1] = X[1, 0], X[1, 1]
        F[c] /= X[2, 2]
        M_buffer[c] = F[c].transpose() @ F[c]
        M_buffer[c] = .5 * (M_buffer[c] - ti.Matrix.identity(ti.f32, 2))
        M_buffer[c] = 2 * miu_S[None] * M_buffer[c] + lambda_S[None] * M_buffer[c].trace() * ti.Matrix.identity(ti.f32, 2)
        M_buffer[c] = F[c] @ M_buffer[c]
        for j in ti.static(range(3)):
            fk[iter_i, triangles[c, j]] += 0.5 * M_buffer[c] @ area_vectors[c, j]

    n = num_particles[None]
    # Compute force
    for i in range(n):
        for j in range(n):
            if connected[i, j] != 0:
                x_ij = xk[iter_i, i] - xk[iter_i, j]
                d = x_ij.normalized()
                # Dashpot damping
                v_rel = (vk[iter_i, i] - vk[iter_i, j]).dot(d)
                fk[iter_i, i] += -dashpot_damping[None] * v_rel * d
@ti.kernel
def substep2():
    n = num_particles[None]
    for i in range(n):
        # Gravity
        f[i] += ti.Vector([0, -9.8]) * m[i]
        xk[0, i] = x[i]
        vk[0, i] = v[i]
        fk[0, i] = f[i]
    substep1(0)
    for i in range(n):
        if not fixed[i]:
            xk[1, i] = x[i] + 0.5 * dt * vk[0, i]
            vk[1, i] = v[i] + 0.5 * dt * fk[0, i] / m[i]
        else:
            xk[1, i] = x[i]
            vk[1, i] = [0, 0]
        bound(1)
        fk[1, i] = f[i]
    substep1(1)
    for i in range(n):
        if not fixed[i]:
            xk[2, i] = x[i] + 0.5 * dt * vk[1, i]
            vk[2, i] = v[i] + 0.5 * dt * fk[1, i] / m[i]
        else:
            xk[2, i] = x[i]
            vk[2, i] = [0, 0]
        bound(2)
        fk[2, i] = f[i]
    substep1(2)
    for i in range(n):
        if not fixed[i]:
            xk[3, i] = x[i] + dt * vk[2, i]
            vk[3, i] = v[i] + dt * fk[2, i] / m[i]
        else:
            xk[3, i] = x[i]
            vk[3, i] = [0, 0]
        bound(3)
        fk[3, i] = f[i]
    substep1(3)
    for i in range(n):
        if not fixed[i]:
            x[i] += dt / 6 * (vk[0, i] + 2 * vk[1, i] + 2 * vk[2, i] + vk[3, i])    
            v[i] += dt / 6 * (fk[0, i] + 2 * fk[1, i] + 2 * fk[2, i] + fk[3, i]) / m[i]
        else:
            v[i] = [0, 0]
        for d in ti.static(range(2)):
            # d = 0: treating X (horizontal) component
            # d = 1: treating Y (vertical) component
            if x[i][d] < 0:  # Bottom and left
                x[i][d] = 0  # move particle inside
                v[i][d] = 0  # stop it from moving further
            if x[i][d] > 1:  # Top and right
                x[i][d] = 1  # move particle inside
                v[i][d] = 0  # stop it from moving further
            
        f[i] = 0
@ti.func 
def bound(iter_i):
    for i in range(num_particles[None]):
        for d in ti.static(range(2)):
            # d = 0: treating X (horizontal) component
            # d = 1: treating Y (vertical) component
            if xk[iter_i, i][d] < 0:  # Bottom and left
                xk[iter_i, i][d] = 0  # move particle inside
                vk[iter_i, i][d] = 0  # stop it from moving further
    
            if xk[iter_i, i][d] > 1:  # Top and right
                xk[iter_i, i][d] = 1  # move particle inside
                vk[iter_i, i][d] = 0  # stop it from moving further

@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32, fixed_: ti.i32):
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    fixed[new_particle_id] = fixed_

    for i in range(num_triangles[None]):
        flag = False
        jdx = -1
        for j in ti.static(range(3)):
            v1 = ti.Vector([pos_x-x[triangles[i, j]][0], pos_y-x[triangles[i, j]][1], 0])
            v2 = ti.Vector([x[triangles[i, (j+1)%3]][0]-x[triangles[i, j]][0], x[triangles[i, (j+1)%3]][1]-x[triangles[i, j]][1], 0])
            if v1.cross(v2)[2] > 0:
                if flag:
                    jdx = -1
                else:
                    flag = True
                    jdx = j
        if flag and jdx >= 0:
            idx1 = triangles[i, jdx]
            idx2 = triangles[i, (jdx + 1) % 3]
            connection_radius = 0.15
            if (x[new_particle_id] - x[idx1]).norm() < connection_radius and (x[new_particle_id] - x[idx2]).norm() < connection_radius:
                # Connect the new particle with particle i
                build_triangle(idx1, idx2, new_particle_id)
                add_line(idx1, new_particle_id)
                add_line(idx2, new_particle_id)
                num_particles[None] += 1

@ti.kernel
def attract(pos_x: ti.f32, pos_y: ti.f32):
    for i in range(num_particles[None]):
        p = ti.Vector([pos_x, pos_y])
        f[i] -= (x[i] - p) * 4


def main():
    gui = ti.GUI("Explicit Mass Spring System", res=(512, 512), background_color=0xDDDDDD)

    drag_damping[None] = 1
    dashpot_damping[None] = 0.2
    num_triangles[None] = 0
    Epsilon_Y = 100.0
    niu_P = 0.2
    generate_obj()
    frame = 0
    while True:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                sys.exit()
            elif e.key == gui.SPACE:
                paused[None] = not paused[None]
            elif e.key == ti.GUI.LMB:
                new_particle(e.pos[0], e.pos[1], int(gui.is_pressed(ti.GUI.SHIFT)))
            elif e.key == "c":
                num_particles[None] = 0
                num_triangles[None] = 0
                connected.fill(0)
                generate_obj()
            elif e.key == "x":
                if gui.is_pressed("Shift"):
                    dashpot_damping[None] /= 1.1
                else:
                    dashpot_damping[None] *= 1.1
            elif e.key == "y":
                if gui.is_pressed("Shift"):
                    Epsilon_Y /= 1.1
                else:
                    Epsilon_Y *= 1.1
            elif e.key == "p":
                if gui.is_pressed("Shift"):
                    niu_P /= 1.1
                else:
                    niu_P *= 1.1
                    
        lambda_S[None] = Epsilon_Y * niu_P / ((1 + niu_P) * (1 - 2 * niu_P))
        miu_S[None] = Epsilon_Y / (2 * (1 + niu_P))
        
        if gui.is_pressed(ti.GUI.RMB):
            cursor_pos = gui.get_cursor_pos()
            attract(cursor_pos[0], cursor_pos[1])
            gui.circle(pos=cursor_pos, color=0x333333, radius=20)

        if not paused[None]:
            for step in range(substeps):
                substep2()
        X = x.to_numpy()
        n = num_particles[None]
                
        # Draw the springs
        for i in range(n):
            for j in range(i + 1, n):
                if connected[i, j] != 0:
                    gui.line(begin=X[i], end=X[j], radius=2, color=0x444444)

        # Draw the particles
        for i in range(n):
            c = 0xFF0000 if fixed[i] else 0x111111
            gui.circle(pos=X[i], color=c, radius=5)

        gui.text(
            content="Left click: add mass point (with shift to fix); Right click: attract",
            pos=(0, 0.99),
            color=0x0,
        )
        gui.text(content="C: clear all; Space: pause", pos=(0, 0.95), color=0x0)
        gui.text(
            content=f"Y: Young's modulus {int(Epsilon_Y)}",
            pos=(0, 0.85),
            color=0x0,
        )
        gui.text(
            content=f"P: Poisson's ratio {niu_P:.3f}",
            pos=(0, 0.8),
            color=0x0,
        )
        # gui.show(f"frames/frame_{frame:05d}.png")
        gui.show()
        frame += 1
        
#ffmpeg -framerate 24 -i frames/frame_%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p fixed_points.mp4

if __name__ == "__main__":
    main()
