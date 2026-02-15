import numpy as np
import plotly.graph_objects as go

L1, L2, L3 = 1.0, 0.8, 0.6
TOL = 1e-3
GAIN = 0.5
MAX_ITER = 300
THETA = np.array([np.pi / 6, np.pi / 4, np.pi / 6])

xd   = float(input("Enter desired x: "))
yd   = float(input("Enter desired y: "))
phid = float(input("Enter desired approach angle (rad): "))

target = np.array([xd, yd])

reach_max = L1 + L2 + L3
reach_min = max(L1 - L2 - L3, 0)
dist = np.linalg.norm(target)

if dist > reach_max or dist < reach_min:
    print(f"Target out of reach. Distance {dist:.4f} not in [{reach_min:.4f}, {reach_max:.4f}]")
    exit()


def jacobian(t):
    t1, t2, t3 = t
    s1   = np.sin(t1)
    s12  = np.sin(t1 + t2)
    s123 = np.sin(t1 + t2 + t3)
    c1   = np.cos(t1)
    c12  = np.cos(t1 + t2)
    c123 = np.cos(t1 + t2 + t3)
    J = np.array([
        [-L1*s1 - L2*s12 - L3*s123,  -L2*s12 - L3*s123,  -L3*s123],
        [ L1*c1 + L2*c12 + L3*c123,   L2*c12 + L3*c123,   L3*c123],
        [ 1.0,                         1.0,                 1.0    ]
    ])
    return J


def forward_kinematics(t):
    t1, t2, t3 = t
    x1 = L1 * np.cos(t1)
    y1 = L1 * np.sin(t1)
    x2 = x1 + L2 * np.cos(t1 + t2)
    y2 = y1 + L2 * np.sin(t1 + t2)
    x3 = x2 + L3 * np.cos(t1 + t2 + t3)
    y3 = y2 + L3 * np.sin(t1 + t2 + t3)
    return np.array([[0, x1, x2, x3], [0, y1, y2, y3]])


def end_effector_pose(t):
    pos = forward_kinematics(t)
    phi = t[0] + t[1] + t[2]
    return pos[:, -1], phi


theta = THETA.copy()
frames = []
frame_labels = []

frames.append(forward_kinematics(theta).copy())
frame_labels.append("Start")

for i in range(MAX_ITER):
    end_pos, phi_cur = end_effector_pose(theta)
    ex   = xd - end_pos[0]
    ey   = yd - end_pos[1]
    ephi = phid - phi_cur
    ephi = (ephi + np.pi) % (2 * np.pi) - np.pi
    err  = np.array([ex, ey, ephi])

    if np.linalg.norm(err) < TOL:
        print(f"Converged at iteration {i}")
        break

    J      = jacobian(theta)
    dtheta = GAIN * np.linalg.pinv(J) @ err
    theta  = theta + dtheta

    frames.append(forward_kinematics(theta).copy())
    frame_labels.append(f"Iter {i + 1}")
else:
    print(f"Did not converge within {MAX_ITER} iterations")

pad   = reach_max * 1.2
W     = "#ffffff"
W_DIM = "rgba(255,255,255,0.25)"

plotly_frames = []
for idx, fpos in enumerate(frames):
    frame_traces = []
    for j in range(3):
        frame_traces.append(go.Scatter(
            x=fpos[0, j:j+2],
            y=fpos[1, j:j+2],
            mode="lines",
            line=dict(color=W, width=3),
            showlegend=False
        ))
    frame_traces.append(go.Scatter(
        x=fpos[0],
        y=fpos[1],
        mode="markers",
        marker=dict(color=W, size=10, symbol="circle",
                    line=dict(color="#000", width=1.5)),
        showlegend=False
    ))
    frame_traces.append(go.Scatter(
        x=[fpos[0, -1]],
        y=[fpos[1, -1]],
        mode="markers",
        marker=dict(color=W, size=14, symbol="circle",
                    line=dict(color="#000", width=2)),
        showlegend=False
    ))
    frame_traces.append(go.Scatter(
        x=[target[0]],
        y=[target[1]],
        mode="markers",
        marker=dict(color=W, size=16, symbol="x",
                    line=dict(color=W, width=3)),
        showlegend=False
    ))
    plotly_frames.append(go.Frame(data=frame_traces, name=str(idx)))

init_pos = frames[0]
init_traces = []
for j in range(3):
    init_traces.append(go.Scatter(
        x=init_pos[0, j:j+2],
        y=init_pos[1, j:j+2],
        mode="lines",
        line=dict(color=W, width=3),
        name=f"Link {j+1}"
    ))
init_traces.append(go.Scatter(
    x=init_pos[0],
    y=init_pos[1],
    mode="markers",
    marker=dict(color=W, size=10, symbol="circle",
                line=dict(color="#000", width=1.5)),
    name="Joints"
))
init_traces.append(go.Scatter(
    x=[init_pos[0, -1]],
    y=[init_pos[1, -1]],
    mode="markers",
    marker=dict(color=W, size=14, symbol="circle",
                line=dict(color="#000", width=2)),
    name="End Effector"
))
init_traces.append(go.Scatter(
    x=[target[0]],
    y=[target[1]],
    mode="markers",
    marker=dict(color=W, size=16, symbol="x",
                line=dict(color=W, width=3)),
    name="Target"
))

fig = go.Figure(
    data=init_traces,
    frames=plotly_frames,
    layout=go.Layout(
        title=dict(
            text=f"3-Link Planar Arm IK  |  Target: ({xd}, {yd})  φ={phid:.3f} rad",
            font=dict(color=W, size=18, family="monospace"),
            x=0.5
        ),
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        xaxis=dict(
            range=[-pad, pad],
            zeroline=True,
            zerolinecolor=W_DIM,
            gridcolor=W_DIM,
            tickfont=dict(color=W),
            title=dict(text="X", font=dict(color=W)),
            scaleanchor="y"
        ),
        yaxis=dict(
            range=[-pad, pad],
            zeroline=True,
            zerolinecolor=W_DIM,
            gridcolor=W_DIM,
            tickfont=dict(color=W),
            title=dict(text="Y", font=dict(color=W))
        ),
        legend=dict(
            font=dict(color=W),
            bgcolor="#000000",
            bordercolor=W,
            borderwidth=1
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0,
            x=0.5,
            xanchor="center",
            yanchor="top",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=40, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0)
                    )]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0)
                    )]
                )
            ],
            font=dict(color="#000000"),
            bgcolor=W,
            bordercolor=W
        )],
        sliders=[dict(
            steps=[
                dict(
                    method="animate",
                    args=[[str(k)], dict(
                        mode="immediate",
                        frame=dict(duration=40, redraw=True),
                        transition=dict(duration=0)
                    )],
                    label=frame_labels[k]
                )
                for k in range(len(plotly_frames))
            ],
            active=0,
            x=0.05,
            len=0.9,
            y=-0.05,
            currentvalue=dict(
                font=dict(color=W, size=12),
                prefix="Step: ",
                visible=True,
                xanchor="center"
            ),
            tickcolor=W,
            font=dict(color=W),
            bgcolor="#000000",
            bordercolor=W
        )]
    )
)

fig.write_html("arm_sim.html")
print("Saved to arm_sim.html")
fig.show()
