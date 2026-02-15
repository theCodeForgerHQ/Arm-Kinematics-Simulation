import numpy as np
import plotly.graph_objects as go

l1, l2, l3 = 1.0, 0.8, 0.6
tol = 1e-4
gain = 0.5
max_iter = 500
theta = np.array([0.5, 0.0, 0.0])

x_des = float(input("Enter desired x: "))
y_des = float(input("Enter desired y: "))
phi_des = float(input("Enter desired phi (in radians): "))


def forward_kinematics(theta):
    t1, t2, t3 = theta
    x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2) + l3 * np.cos(t1 + t2 + t3)
    y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2) + l3 * np.sin(t1 + t2 + t3)
    phi = t1 + t2 + t3
    return x, y, phi


def jacobian(theta):
    t1, t2, t3 = theta
    s1 = np.sin(t1)
    s12 = np.sin(t1 + t2)
    s123 = np.sin(t1 + t2 + t3)
    c1 = np.cos(t1)
    c12 = np.cos(t1 + t2)
    c123 = np.cos(t1 + t2 + t3)

    J = np.array([
        [
            -l1 * s1 - l2 * s12 - l3 * s123,
            -l2 * s12 - l3 * s123,
            -l3 * s123
        ],
        [
            l1 * c1 + l2 * c12 + l3 * c123,
            l2 * c12 + l3 * c123,
            l3 * c123
        ],
        [1.0, 1.0, 1.0]
    ])
    return J


r = np.sqrt(x_des**2 + y_des**2)

if r > l1 + l2 + l3:
    print(f"Target unreachable: distance {r:.4f} exceeds total arm length {l1 + l2 + l3:.4f}.")
    exit()

if r < max(l1 - l2 - l3, 0):
    print(f"Target unreachable: distance {r:.4f} is less than minimum reach {max(l1 - l2 - l3, 0):.4f}.")
    exit()


def get_joint_positions(theta):
    t1, t2, t3 = theta
    p0 = np.array([0.0, 0.0])
    p1 = p0 + np.array([l1 * np.cos(t1), l1 * np.sin(t1)])
    p2 = p1 + np.array([l2 * np.cos(t1 + t2), l2 * np.sin(t1 + t2)])
    p3 = p2 + np.array([l3 * np.cos(t1 + t2 + t3), l3 * np.sin(t1 + t2 + t3)])
    return p0, p1, p2, p3


frame_data = []
converged = False

for i in range(max_iter):
    x_cur, y_cur, phi_cur = forward_kinematics(theta)

    error = np.array([x_des - x_cur, y_des - y_cur, phi_des - phi_cur])

    p0, p1, p2, p3 = get_joint_positions(theta)
    frame_data.append({
        "joints": [p0, p1, p2, p3],
        "error_norm": np.linalg.norm(error),
        "iteration": i
    })

    if np.linalg.norm(error) < tol:
        converged = True
        break

    J = jacobian(theta)
    J_pinv = np.linalg.pinv(J)
    delta_theta = gain * J_pinv @ error
    theta = theta + delta_theta

if not converged:
    print(f"Did not converge within {max_iter} iterations. Final error: {np.linalg.norm(error):.6f}")
    exit()

print(f"Converged in {len(frame_data)} iterations.")
print(f"Final joint angles: {np.degrees(theta)} degrees")

total_length = l1 + l2 + l3
axis_range = [-total_length * 1.3, total_length * 1.3]

step = max(1, len(frame_data) // 80)
sampled_indices = list(range(0, len(frame_data), step))
if (len(frame_data) - 1) not in sampled_indices:
    sampled_indices.append(len(frame_data) - 1)
sampled_frames = [frame_data[i] for i in sampled_indices]


def make_arm_traces(frame):
    joints = frame["joints"]
    p0, p1, p2, p3 = joints

    link_x = [p0[0], p1[0], None, p1[0], p2[0], None, p2[0], p3[0]]
    link_y = [p0[1], p1[1], None, p1[1], p2[1], None, p2[1], p3[1]]

    link_trace = go.Scatter(
        x=link_x,
        y=link_y,
        mode="lines",
        line=dict(color="#1a1a2e", width=3),
        name="Links",
        showlegend=False
    )

    joint_x = [p0[0], p1[0], p2[0]]
    joint_y = [p0[1], p1[1], p2[1]]

    joint_trace = go.Scatter(
        x=joint_x,
        y=joint_y,
        mode="markers",
        marker=dict(color="#16213e", size=10, symbol="circle",
                    line=dict(color="#e94560", width=2)),
        name="Joints",
        showlegend=False
    )

    ee_trace = go.Scatter(
        x=[p3[0]],
        y=[p3[1]],
        mode="markers",
        marker=dict(color="#e94560", size=12, symbol="circle",
                    line=dict(color="#ffffff", width=2)),
        name="End-Effector",
        showlegend=True
    )

    return [link_trace, joint_trace, ee_trace]


target_trace = go.Scatter(
    x=[x_des],
    y=[y_des],
    mode="markers",
    marker=dict(color="#0f3460", size=14, symbol="x",
                line=dict(color="#0f3460", width=3)),
    name="Target",
    showlegend=True
)

initial_arm_traces = make_arm_traces(sampled_frames[0])

all_initial_traces = initial_arm_traces + [target_trace]

frames = []
for idx, frame in enumerate(sampled_frames):
    arm_traces = make_arm_traces(frame)
    frame_traces = arm_traces + [target_trace]
    frames.append(go.Frame(
        data=frame_traces,
        name=str(idx)
    ))

fig = go.Figure(
    data=all_initial_traces,
    frames=frames
)

fig.update_layout(
    title=dict(
        text="3-Link Planar Arm — Inverse Kinematics",
        font=dict(size=18, color="#1a1a2e"),
        x=0.5,
        xanchor="center"
    ),
    xaxis=dict(
        range=axis_range,
        scaleanchor="y",
        scaleratio=1,
        showgrid=True,
        gridcolor="#e8e8e8",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="#cccccc",
        zerolinewidth=1,
        showline=True,
        linecolor="#cccccc",
        tickfont=dict(size=11, color="#555555"),
        title=dict(text="x", font=dict(size=13, color="#333333"))
    ),
    yaxis=dict(
        range=axis_range,
        showgrid=True,
        gridcolor="#e8e8e8",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="#cccccc",
        zerolinewidth=1,
        showline=True,
        linecolor="#cccccc",
        tickfont=dict(size=11, color="#555555"),
        title=dict(text="y", font=dict(size=13, color="#333333"))
    ),
    plot_bgcolor="#fafafa",
    paper_bgcolor="#ffffff",
    legend=dict(
        font=dict(size=12, color="#333333"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#dddddd",
        borderwidth=1,
        x=0.02,
        y=0.98
    ),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            y=1.12,
            x=0.5,
            xanchor="center",
            buttons=[
                dict(
                    label="▶  Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=40, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=20, easing="linear")
                        )
                    ]
                ),
                dict(
                    label="⏸  Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0)
                        )
                    ]
                )
            ],
            font=dict(size=12, color="#1a1a2e"),
            bgcolor="#f0f0f0",
            bordercolor="#cccccc"
        )
    ],
    sliders=[
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[[str(k)], dict(
                        frame=dict(duration=0, redraw=True),
                        mode="immediate",
                        transition=dict(duration=0)
                    )],
                    label=str(k)
                )
                for k in range(len(sampled_frames))
            ],
            transition=dict(duration=0),
            x=0.0,
            y=0,
            currentvalue=dict(
                font=dict(size=11, color="#555555"),
                prefix="Frame: ",
                visible=True,
                xanchor="center"
            ),
            len=1.0,
            bgcolor="#f0f0f0",
            bordercolor="#cccccc"
        )
    ],
    margin=dict(l=60, r=40, t=100, b=80),
)

import webbrowser, os
output_path = os.path.abspath("ik_arm_convergence_simulation.html")
fig.write_html(output_path, auto_play=False)
webbrowser.open(f"file://{output_path}")
print(f"Simulation saved and opened: {output_path}")

theta_start = np.array([0.5, 0.0, 0.0])
theta_end_raw = theta.copy()

theta_end_wrapped = np.arctan2(np.sin(theta_end_raw), np.cos(theta_end_raw))
theta_start_wrapped = np.arctan2(np.sin(theta_start), np.cos(theta_start))

delta = theta_end_wrapped - theta_start_wrapped
delta = (delta + np.pi) % (2 * np.pi) - np.pi
theta_end = theta_start_wrapped + delta

movj_frames = 120
t_raw = np.linspace(0.0, 1.0, movj_frames)
t_smooth = t_raw * t_raw * (3.0 - 2.0 * t_raw)

movj_thetas = np.array([
    theta_start_wrapped + s * delta
    for s in t_smooth
])

movj_ee_trail_x = []
movj_ee_trail_y = []

start_joints = get_joint_positions(theta_start)

movj_start_marker = go.Scatter(
    x=[start_joints[3][0]],
    y=[start_joints[3][1]],
    mode="markers",
    marker=dict(color="#aaaaaa", size=10, symbol="circle-open",
                line=dict(color="#aaaaaa", width=2)),
    name="Start",
    showlegend=True
)

movj_target_trace = go.Scatter(
    x=[x_des],
    y=[y_des],
    mode="markers",
    marker=dict(color="#0f3460", size=14, symbol="x",
                line=dict(color="#0f3460", width=3)),
    name="Target",
    showlegend=True
)


def make_movj_arm_traces(joints, ee_trail_x=None, ee_trail_y=None):
    p0, p1, p2, p3 = joints

    link_x = [p0[0], p1[0], None, p1[0], p2[0], None, p2[0], p3[0]]
    link_y = [p0[1], p1[1], None, p1[1], p2[1], None, p2[1], p3[1]]

    link_trace = go.Scatter(
        x=link_x,
        y=link_y,
        mode="lines",
        line=dict(color="#1a1a2e", width=3),
        name="Links",
        showlegend=False
    )

    joint_x = [p0[0], p1[0], p2[0]]
    joint_y = [p0[1], p1[1], p2[1]]

    joint_trace = go.Scatter(
        x=joint_x,
        y=joint_y,
        mode="markers",
        marker=dict(color="#16213e", size=10, symbol="circle",
                    line=dict(color="#e94560", width=2)),
        name="Joints",
        showlegend=False
    )

    ee_trace = go.Scatter(
        x=[p3[0]],
        y=[p3[1]],
        mode="markers",
        marker=dict(color="#e94560", size=12, symbol="circle",
                    line=dict(color="#ffffff", width=2)),
        name="End-Effector",
        showlegend=True
    )

    if ee_trail_x is not None and len(ee_trail_x) > 1:
        trail_trace = go.Scatter(
            x=list(ee_trail_x),
            y=list(ee_trail_y),
            mode="lines",
            line=dict(color="#e94560", width=1, dash="dot"),
            opacity=0.45,
            name="EE Trail",
            showlegend=False
        )
        return [link_trace, joint_trace, ee_trace, trail_trace]

    return [link_trace, joint_trace, ee_trace]


movj_initial_joints = get_joint_positions(movj_thetas[0])
movj_initial_arm = make_movj_arm_traces(movj_initial_joints, None, None)
movj_initial_data = movj_initial_arm + [movj_start_marker, movj_target_trace]

movj_plotly_frames = []
for idx, th in enumerate(movj_thetas):
    joints = get_joint_positions(th)
    movj_ee_trail_x.append(joints[3][0])
    movj_ee_trail_y.append(joints[3][1])
    arm_traces = make_movj_arm_traces(joints, movj_ee_trail_x[:], movj_ee_trail_y[:])
    frame_traces = arm_traces + [movj_start_marker, movj_target_trace]
    movj_plotly_frames.append(go.Frame(
        data=frame_traces,
        name=str(idx)
    ))

fig2 = go.Figure(
    data=movj_initial_data,
    frames=movj_plotly_frames
)

fig2.update_layout(
    title=dict(
        text="3-Link Planar Arm — MOVJ Joint Interpolation",
        font=dict(size=18, color="#1a1a2e"),
        x=0.5,
        xanchor="center"
    ),
    xaxis=dict(
        range=axis_range,
        scaleanchor="y",
        scaleratio=1,
        showgrid=True,
        gridcolor="#e8e8e8",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="#cccccc",
        zerolinewidth=1,
        showline=True,
        linecolor="#cccccc",
        tickfont=dict(size=11, color="#555555"),
        title=dict(text="x", font=dict(size=13, color="#333333"))
    ),
    yaxis=dict(
        range=axis_range,
        showgrid=True,
        gridcolor="#e8e8e8",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="#cccccc",
        zerolinewidth=1,
        showline=True,
        linecolor="#cccccc",
        tickfont=dict(size=11, color="#555555"),
        title=dict(text="y", font=dict(size=13, color="#333333"))
    ),
    plot_bgcolor="#fafafa",
    paper_bgcolor="#ffffff",
    legend=dict(
        font=dict(size=12, color="#333333"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#dddddd",
        borderwidth=1,
        x=0.02,
        y=0.98
    ),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            y=1.12,
            x=0.5,
            xanchor="center",
            buttons=[
                dict(
                    label="▶  Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=33, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=16, easing="linear")
                        )
                    ]
                ),
                dict(
                    label="⏸  Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0)
                        )
                    ]
                )
            ],
            font=dict(size=12, color="#1a1a2e"),
            bgcolor="#f0f0f0",
            bordercolor="#cccccc"
        )
    ],
    sliders=[
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[[str(k)], dict(
                        frame=dict(duration=0, redraw=True),
                        mode="immediate",
                        transition=dict(duration=0)
                    )],
                    label=str(k)
                )
                for k in range(movj_frames)
            ],
            transition=dict(duration=0),
            x=0.0,
            y=0,
            currentvalue=dict(
                font=dict(size=11, color="#555555"),
                prefix="Frame: ",
                visible=True,
                xanchor="center"
            ),
            len=1.0,
            bgcolor="#f0f0f0",
            bordercolor="#cccccc"
        )
    ],
    margin=dict(l=60, r=40, t=100, b=80),
)

import webbrowser, os
movj_output_path = os.path.abspath("movj_arm_simulation.html")
fig2.write_html(movj_output_path, auto_play=False)
webbrowser.open(f"file://{movj_output_path}")
print(f"MOVJ simulation saved and opened: {movj_output_path}")