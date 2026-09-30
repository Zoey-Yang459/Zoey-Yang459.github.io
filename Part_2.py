import matplotlib.pyplot as plt
from pathlib import Path

# Get the directory containing this Python file
BASE_DIR = Path(__file__).resolve().parent

# Create/find the plots folder inside the assignment folder
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# Given data points
points = [
    (0, 0.5),
    (2, 3.5),
    (1, 1.5),
    (3, 7.5)
]


# try y = mx + b


def line_mse(m, b, points):
    total_error = 0

    for x, y in points:
        y_pred = m * x + b
        total_error += (y_pred - y) ** 2

    return total_error / len(points)


def line_newton(
    points,
    m0=0.0,
    b0=0.0,
    tolerance=1e-7,
    max_iter=100
):
    n = len(points)

    m = m0
    b = b0

    history = [
        (0, m, b, line_mse(m, b, points))
    ]

    for iteration in range(1, max_iter + 1):

        dMSE_dm = (
            2 / n
        ) * sum(
            x * (m * x + b - y)
            for x, y in points
        )

        d2MSE_dm2 = (
            2 / n
        ) * sum(
            x ** 2
            for x, y in points
        )

        new_m = m - dMSE_dm / d2MSE_dm2


        dMSE_db = (
            2 / n
        ) * sum(
            new_m * x + b - y
            for x, y in points
        )

        d2MSE_db2 = 2

        new_b = b - dMSE_db / d2MSE_db2


        # Calculate new MSE
        mse = line_mse(new_m, new_b, points)

        history.append(
            (iteration, new_m, new_b, mse)
        )


        # Check convergence
        if (
            abs(new_m - m) < tolerance
            and abs(new_b - b) < tolerance
        ):
            m = new_m
            b = new_b
            break


        m = new_m
        b = new_b


    return m, b, line_mse(m, b, points), history



# RUN LINE NEWTON METHOD

line_m, line_b, line_final_mse, line_history = line_newton(
    points,
    m0=0,
    b0=0
)

print("LINE FIT")

for iteration, m, b, mse in line_history:
    print(
        f"Iteration {iteration}: "
        f"m = {m:.6f}, "
        f"b = {b:.6f}, "
        f"MSE = {mse:.6f}"
    )


print("\nFinal line:")
print(
    f"y = {line_m:.6f}x + ({line_b:.6f})"
)

print(
    f"Final MSE = {line_final_mse:.6f}"
)


# y = ax^2 + bx + c

def parabola_mse(a, b, c, points):
    total_error = 0

    for x, y in points:
        y_pred = a * x**2 + b * x + c
        total_error += (y_pred - y) ** 2

    return total_error / len(points)



def parabola_newton(
    points,
    a0=0.0,
    b0=0.0,
    c0=0.0,
    tolerance=1e-7,
    max_iter=1000
):
    n = len(points)

    a = a0
    b = b0
    c = c0

    history = [
        (
            0,
            a,
            b,
            c,
            parabola_mse(a, b, c, points)
        )
    ]


    for iteration in range(1, max_iter + 1):

        dMSE_da = (
            2 / n
        ) * sum(
            x**2 *
            (a * x**2 + b * x + c - y)
            for x, y in points
        )

        d2MSE_da2 = (
            2 / n
        ) * sum(
            x**4
            for x, y in points
        )

        new_a = a - dMSE_da / d2MSE_da2


        dMSE_db = (
            2 / n
        ) * sum(
            x *
            (new_a * x**2 + b * x + c - y)
            for x, y in points
        )

        d2MSE_db2 = (
            2 / n
        ) * sum(
            x**2
            for x, y in points
        )

        new_b = b - dMSE_db / d2MSE_db2

        dMSE_dc = (
            2 / n
        ) * sum(
            new_a * x**2 +
            new_b * x +
            c -
            y
            for x, y in points
        )

        d2MSE_dc2 = 2

        new_c = c - dMSE_dc / d2MSE_dc2


        # Calculate new MSE
        mse = parabola_mse(
            new_a,
            new_b,
            new_c,
            points
        )


        history.append(
            (
                iteration,
                new_a,
                new_b,
                new_c,
                mse
            )
        )


        # Check convergence
        if (
            abs(new_a - a) < tolerance
            and abs(new_b - b) < tolerance
            and abs(new_c - c) < tolerance
        ):
            a = new_a
            b = new_b
            c = new_c
            break


        a = new_a
        b = new_b
        c = new_c


    return (
        a,
        b,
        c,
        parabola_mse(a, b, c, points),
        history
    )



# RUN PARABOLA NEWTON METHOD
(
    para_a,
    para_b,
    para_c,
    para_final_mse,
    para_history
) = parabola_newton(
    points,
    a0=0,
    b0=0,
    c0=0
)

print("PARABOLA FIT")

# Only show the first few iterations
for iteration, a, b, c, mse in para_history[:10]:
    print(
        f"Iteration {iteration}: "
        f"a = {a:.6f}, "
        f"b = {b:.6f}, "
        f"c = {c:.6f}, "
        f"MSE = {mse:.6f}"
    )


print("\nFinal parabola:")

print(
    f"y = {para_a:.6f}x^2 "
    f"+ ({para_b:.6f})x "
    f"+ ({para_c:.6f})"
)

print(
    f"Final MSE = {para_final_mse:.6f}"
)



x_values = [
    -0.5 + i * 0.01
    for i in range(401)
]


plt.figure(figsize=(9, 6))


# Plot original data points
for x, y in points:
    plt.scatter(x, y, s=80)


# Plot first few Newton iterations
iterations_to_plot = [
    0,
    1,
    2,
    3
]


for index in iterations_to_plot:

    iteration, m, b, mse = line_history[index]

    y_values = [
        m * x + b
        for x in x_values
    ]

    plt.plot(
        x_values,
        y_values,
        label=(
            f"Iter {iteration}: "
            f"m={m:.3f}, b={b:.3f}"
        )
    )


# Plot final line
final_y_line = [
    line_m * x + line_b
    for x in x_values
]


plt.plot(
    x_values,
    final_y_line,
    linestyle="--",
    linewidth=2,
    label=(
        f"Final: y={line_m:.3f}x"
        f"{line_b:+.3f}"
    )
)


plt.title(
    "Newton-Raphson Intermediate Steps: Line Fit"
)

plt.xlabel("x")
plt.ylabel("y")

plt.grid(True)
plt.legend()

plt.savefig(
    PLOTS_DIR / "line_newton_iterations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# Skip iteration 0 so the convergence is visible
line_iterations = [
    item[0]
    for item in line_history[1:]
]

line_mse_values = [
    item[3]
    for item in line_history[1:]
]

plt.figure(figsize=(8, 5))

plt.plot(
    line_iterations,
    line_mse_values,
    marker="o"
)

plt.axhline(
    y=0.575,
    linestyle="--",
    label="Analytical minimum MSE = 0.575"
)

plt.title("Line Fit: MSE Convergence")

plt.xlabel("Iteration")
plt.ylabel("MSE")

plt.grid(True)
plt.legend()

plt.savefig(
    PLOTS_DIR / "line_mse_convergence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


plt.figure(figsize=(9, 6))


# Plot data points
for x, y in points:
    plt.scatter(x, y, s=80)


# First few parabola iterations
iterations_to_plot = [
    0,
    1,
    2,
    3
]


for index in iterations_to_plot:

    (
        iteration,
        a,
        b,
        c,
        mse
    ) = para_history[index]


    y_values = [
        a * x**2 + b * x + c
        for x in x_values
    ]


    plt.plot(
        x_values,
        y_values,
        label=(
            f"Iter {iteration}: "
            f"a={a:.3f}, "
            f"b={b:.3f}, "
            f"c={c:.3f}"
        )
    )


# Final parabola
final_y_parabola = [
    para_a * x**2
    + para_b * x
    + para_c
    for x in x_values
]


plt.plot(
    x_values,
    final_y_parabola,
    linestyle="--",
    linewidth=2,
    label=(
        "Final parabola"
    )
)


plt.title(
    "Newton-Raphson Intermediate Steps: Parabola Fit"
)

plt.xlabel("x")
plt.ylabel("y")

plt.grid(True)
plt.legend()

plt.savefig(
    PLOTS_DIR / "parabola_newton_iterations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Skip iteration 0 and display the early convergence clearly
para_plot_history = para_history[1:16]

para_iterations = [
    item[0]
    for item in para_plot_history
]

para_mse_values = [
    item[4]
    for item in para_plot_history
]

plt.figure(figsize=(8, 5))

plt.plot(
    para_iterations,
    para_mse_values,
    marker="o"
)

plt.axhline(
    y=0.0125,
    linestyle="--",
    label="Analytical minimum MSE = 0.0125"
)

plt.title("Parabola Fit: MSE Convergence")

plt.xlabel("Iteration")
plt.ylabel("MSE")

plt.grid(True)
plt.legend()

plt.savefig(
    PLOTS_DIR / "parabola_mse_convergence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
