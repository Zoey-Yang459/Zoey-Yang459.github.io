import math
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return x**2 + 5

# First derivative
def df(x):
    return 2 * x

# Second derivative
def ddf(x):
    return 2

# Second parabola for testing
def f2(x):
    return 0.5 * x**2 - 2 * x + 3

def df2(x):
    return x - 2

def ddf2(x):
    return 1

# Non-polynomial function: exponential
def f3(x):
    return math.exp(x)

def df3(x):
    return math.exp(x)

def ddf3(x):
    return math.exp(x)



#Points: 
# [0,0],[-4,0],[-8,0],[2,0],[6,0]

def find_newton_distance(
    x0,
    y0,
    f,
    df,
    ddf,
    initial_guess=0.0,
    tolerance=1e-7,
    max_iter=100
    ):

    x = initial_guess

    history = [x]

    for _ in range(max_iter):
        # D'(x)
        D_prime = 2 * (x-x0) + 2 * (f(x) - y0) * df(x)

        #D''(x)
        D_double_prime = 2 + 2 * (df(x)**2) + 2 * (f(x) - y0) * ddf(x)

        #Newton-Raphson update step
        next_x = x - D_prime / D_double_prime

        history.append(next_x)

        if abs (next_x - x) < tolerance:
            x = next_x
            break
        
        x = next_x
    
    shortest_distance = ((x - x0)**2 + (f(x) - y0)**2)**0.5

    return shortest_distance, x, history

def golden_section_search(
    x0,
    y0,
    f,
    a,
    b,
    tolerance = 1e-7
):
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)

    def dist_sq(x):
        return (x - x0)**2 + (f(x) - y0)**2

    f_x1 = dist_sq(x1)
    f_x2 = dist_sq(x2)

    history = []

    while abs(b - a) > tolerance:

        history.append((a, b, x1, x2, f_x1, f_x2))
        
        if f_x1 < f_x2:
            b = x2
            x2 = x1
            f_x2 = f_x1
            x1 = a + resphi * (b - a)
            f_x1 = dist_sq(x1)
        else:
            a = x1
            x1 = x2
            f_x1 = f_x2
            x2 = b - resphi * (b - a)
            f_x2 = dist_sq(x2)

    

    best_x = (a + b)/2
    return math.sqrt(dist_sq(best_x)), best_x, history

def plot_newton_steps(
    x0,
    y0,
    f,
    df,
    ddf,
    initial_guess,
    x_min,
    x_max,
    equation_label,
    filename
):
    distance, closest_x, history = find_newton_distance(
        x0,
        y0,
        f,
        df,
        ddf,
        initial_guess=initial_guess
    )

    x_values = [
        x_min + i * 0.01
        for i in range(int((x_max - x_min) / 0.01) + 1)
    ]

    y_values = [f(x) for x in x_values]

    plt.figure(figsize=(9, 6))

    plt.plot(
        x_values,
        y_values,
        label=equation_label
    )

    plt.scatter(
        x0,
        y0,
        s=80,
        label=f"Given point ({x0}, {y0})"
    )

    # Show initial guess + first 4 Newton steps
    for i, x_iter in enumerate(history[:5]):
        y_iter = f(x_iter)

        plt.scatter(x_iter, y_iter)

        plt.annotate(
            f"Iter {i}",
            (x_iter, y_iter),
            textcoords="offset points",
            xytext=(6, 6)
        )

    closest_y = f(closest_x)

    plt.scatter(
        closest_x,
        closest_y,
        s=100,
        label="Closest point"
    )

    plt.plot(
        [x0, closest_x],
        [y0, closest_y],
        linestyle="--",
        label=f"Shortest distance = {distance:.3f}"
    )

    plt.title(
        f"Newton-Raphson Method for Point ({x0}, {y0})"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_golden_steps(
    x0,
    y0,
    f,
    a,
    b,
    x_min,
    x_max,
    equation_label,
    filename
):
    distance, closest_x, history = golden_section_search(
        x0,
        y0,
        f,
        a,
        b
    )

    x_values = [
        x_min + i * 0.01
        for i in range(int((x_max - x_min) / 0.01) + 1)
    ]

    y_values = [f(x) for x in x_values]

    plt.figure(figsize=(9, 6))

    plt.plot(
        x_values,
        y_values,
        label=equation_label
    )

    plt.scatter(
        x0,
        y0,
        s=80,
        label=f"Given point ({x0}, {y0})"
    )

    # Show first 2 Golden Section iterations
    for i, step in enumerate(history[:2], start=1):
        a_i, b_i, x1, x2, f_x1, f_x2 = step

        y1 = f(x1)
        y2 = f(x2)

        plt.scatter(x1, y1)
        plt.scatter(x2, y2)

        plt.annotate(
            f"Iter {i}: x1",
            (x1, y1),
            textcoords="offset points",
            xytext=(6, 7)
        )

        plt.annotate(
            f"Iter {i}: x2",
            (x2, y2),
            textcoords="offset points",
            xytext=(6, -15)
        )

    closest_y = f(closest_x)

    plt.scatter(
        closest_x,
        closest_y,
        s=100,
        label="Closest point"
    )

    plt.plot(
        [x0, closest_x],
        [y0, closest_y],
        linestyle="--",
        label=f"Shortest distance = {distance:.3f}"
    )

    plt.title(
        f"Golden Section Search for Point ({x0}, {y0})"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# Test all required points
points = [
    (0, 0),
    (-4, 0),
    (-8, 0),
    (2, 0),
    (6, 0)
]

print(f"Newton_Raphson Method:")
for x0, y0 in points:
    distance, closest_x, history = find_newton_distance(
        x0,
        y0,
        f,
        df,
        ddf,
        initial_guess=x0
    )
    
    print(f"Point: ({x0}, {y0})")
    print(f"Shortest distance: {distance:.6f}")
    print(f"Closest x: {closest_x:.6f}")
    print("Iterations:")

    for i in range(len(history) - 1):
        print(
            f"Iteration {i + 1}: "
            f"{history[i]:.6f} -> {history[i + 1]:.6f}"
        )
    
    print()


# Test all required points using Golden Section Search
test_cases = [
    ((0, 0), -1, 1),
    ((-4, 0), -4, 0),
    ((-8, 0), -8, 0),
    ((2, 0), 0, 2),
    ((6, 0), 0, 6)
]

print(f"Golden Section Search Method:")
for (x0, y0), a, b in test_cases:
    distance, closest_x, history = golden_section_search(
        x0,
        y0,
        f,
        a,
        b
    )

    print(f"Point: ({x0}, {y0})")
    print(f"Interval: [{a}, {b}]")
    print(f"Shortest distance: {distance:.6f}")
    print(f"Closest x: {closest_x:.6f}")

    print("Iterations:")

    for i, step in enumerate(history, start=1):
        a_i, b_i, x1, x2, f_x1, f_x2 = step
        print(
            f"Iteration {i}: "
            f"Interval = [{a_i:.6f}, {b_i:.6f}], "
            f"x1 = {x1:.6f}, "
            f"x2 = {x2:.6f}, "
            f"D(x1) = {f_x1:.6f}, "
            f"D(x2) = {f_x2:.6f}"
        )

    print()


print("\nOther Parabola Test:")

# Test point
x0 = 0
y0 = 0

# Newton-Raphson
newton_distance, newton_x, newton_history = find_newton_distance(
    x0,
    y0,
    f2,
    df2,
    ddf2,
    initial_guess=0
)

print("Newton-Raphson:")
print(f"Point: ({x0}, {y0})")
print(f"Closest x: {newton_x:.6f}")
print(f"Shortest distance: {newton_distance:.6f}")


# Golden Section Search
golden_distance, golden_x, golden_history = golden_section_search(
    x0,
    y0,
    f2,
    -5,
    5
)

print("\nGolden Section Search:")
print(f"Point: ({x0}, {y0})")
print(f"Closest x: {golden_x:.6f}")
print(f"Shortest distance: {golden_distance:.6f}")


print("\nNon-Polynomial Function Test: y = e^x")

x0 = 0
y0 = 0

# Newton-Raphson
newton_distance, newton_x, newton_history = find_newton_distance(
    x0,
    y0,
    f3,
    df3,
    ddf3,
    initial_guess=0
)

print("Newton-Raphson:")
print(f"Point: ({x0}, {y0})")
print(f"Closest x: {newton_x:.6f}")
print(f"Shortest distance: {newton_distance:.6f}")

# Golden Section Search
golden_distance, golden_x, golden_history = golden_section_search(
    x0,
    y0,
    f3,
    -2,
    1
)

print("\nGolden Section Search:")
print(f"Point: ({x0}, {y0})")
print(f"Closest x: {golden_x:.6f}")
print(f"Shortest distance: {golden_distance:.6f}")

# Plot for non-polynomial example: y = e^x

x0_plot = 0
y0_plot = 0

distance, closest_x, newton_history = find_newton_distance(
    x0_plot,
    y0_plot,
    f3,
    df3,
    ddf3,
    initial_guess=0
)

# Values for drawing y = e^x
x_values = [-2 + i * 0.01 for i in range(301)]
y_values = [f3(x) for x in x_values]

plt.figure(figsize=(9, 6))

# Plot exponential function
plt.plot(
    x_values,
    y_values,
    label=r"$y=e^x$"
)

# Given point
plt.scatter(
    x0_plot,
    y0_plot,
    s=80,
    label="Given point (0, 0)"
)

# First few Newton iterations
for i, x_iter in enumerate(newton_history[:3]):
    y_iter = f3(x_iter)

    plt.scatter(x_iter, y_iter)

    plt.annotate(
        f"Iter {i}",
        (x_iter, y_iter),
        textcoords="offset points",
        xytext=(7, 7)
    )

# Final closest point
closest_y = f3(closest_x)

plt.scatter(
    closest_x,
    closest_y,
    s=100,
    label="Closest point"
)

# Shortest-distance line
plt.plot(
    [x0_plot, closest_x],
    [y0_plot, closest_y],
    linestyle="--",
    label=f"Shortest distance = {distance:.3f}"
)

plt.title("Newton-Raphson Method for $y=e^x$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.savefig(
    "plots/newton_exp_0_0.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Golden Section Search plot for non-polynomial example: y = e^x

x0_plot = 0
y0_plot = 0

distance, closest_x, golden_history = golden_section_search(
    x0_plot,
    y0_plot,
    f3,
    -2,
    1
)

# Values for drawing y = e^x
x_values = [-2 + i * 0.01 for i in range(301)]
y_values = [f3(x) for x in x_values]

plt.figure(figsize=(9, 6))

# Plot exponential function
plt.plot(
    x_values,
    y_values,
    label=r"$y=e^x$"
)

# Given point
plt.scatter(
    x0_plot,
    y0_plot,
    s=80,
    label="Given point (0, 0)"
)

# Show first 2 Golden Section iterations
for i, step in enumerate(golden_history[:2], start=1):

    a_i, b_i, x1, x2, f_x1, f_x2 = step

    y1 = f3(x1)
    y2 = f3(x2)

    plt.scatter(x1, y1)
    plt.scatter(x2, y2)

    plt.annotate(
        f"Iter {i}: x1",
        (x1, y1),
        textcoords="offset points",
        xytext=(7, 7)
    )

    plt.annotate(
        f"Iter {i}: x2",
        (x2, y2),
        textcoords="offset points",
        xytext=(7, -15)
    )

# Final closest point
closest_y = f3(closest_x)

plt.scatter(
    closest_x,
    closest_y,
    s=100,
    label="Closest point"
)

# Shortest-distance line
plt.plot(
    [x0_plot, closest_x],
    [y0_plot, closest_y],
    linestyle="--",
    label=f"Shortest distance = {distance:.3f}"
)

plt.title("Golden Section Search for $y=e^x$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.savefig(
    "plots/golden_exp_0_0.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plot_newton_steps(
    x0=-8,
    y0=0,
    f=f,
    df=df,
    ddf=ddf,
    initial_guess=-8,
    x_min=-8.5,
    x_max=2,
    equation_label=r"$y=x^2+5$",
    filename="plots/newton_neg8_0_function.png"
)

plot_newton_steps(
    x0=0,
    y0=0,
    f=f,
    df=df,
    ddf=ddf,
    initial_guess=0,
    x_min=-2,
    x_max=2,
    equation_label=r"$y=x^2+5$",
    filename="plots/newton_0_0.png"
)

plot_newton_steps(
    x0=-4,
    y0=0,
    f=f,
    df=df,
    ddf=ddf,
    initial_guess=-4,
    x_min=-4.5,
    x_max=2,
    equation_label=r"$y=x^2+5$",
    filename="plots/newton_neg4_0.png"
)

plot_newton_steps(
    x0=2,
    y0=0,
    f=f,
    df=df,
    ddf=ddf,
    initial_guess=2,
    x_min=-2,
    x_max=2.5,
    equation_label=r"$y=x^2+5$",
    filename="plots/newton_2_0.png"
)

plot_newton_steps(
    x0=6,
    y0=0,
    f=f,
    df=df,
    ddf=ddf,
    initial_guess=6,
    x_min=-2,
    x_max=6.5,
    equation_label=r"$y=x^2+5$",
    filename="plots/newton_6_0.png"
)

plot_golden_steps(
    x0=-8,
    y0=0,
    f=f,
    a=-8,
    b=0,
    x_min=-8.5,
    x_max=2,
    equation_label=r"$y=x^2+5$",
    filename="plots/golden_neg8_0_function.png"
)

plot_golden_steps(
    x0=0,
    y0=0,
    f=f,
    a=-1,
    b=1,
    x_min=-2,
    x_max=2,
    equation_label=r"$y=x^2+5$",
    filename="plots/golden_0_0.png"
)

plot_golden_steps(
    x0=-4,
    y0=0,
    f=f,
    a=-4,
    b=0,
    x_min=-4.5,
    x_max=2,
    equation_label=r"$y=x^2+5$",
    filename="plots/golden_neg4_0.png"
)

plot_golden_steps(
    x0=2,
    y0=0,
    f=f,
    a=0,
    b=2,
    x_min=-2,
    x_max=2.5,
    equation_label=r"$y=x^2+5$",
    filename="plots/golden_2_0.png"
)

plot_golden_steps(
    x0=6,
    y0=0,
    f=f,
    a=0,
    b=6,
    x_min=-2,
    x_max=6.5,
    equation_label=r"$y=x^2+5$",
    filename="plots/golden_6_0.png"
)

plot_newton_steps(
    x0=0,
    y0=0,
    f=f2,
    df=df2,
    ddf=ddf2,
    initial_guess=0,
    x_min=-2,
    x_max=5,
    equation_label=r"$y=0.5x^2-2x+3$",
    filename="plots/newton_second_parabola.png"
)

plot_golden_steps(
    x0=0,
    y0=0,
    f=f2,
    a=-5,
    b=5,
    x_min=-2,
    x_max=5,
    equation_label=r"$y=0.5x^2-2x+3$",
    filename="plots/golden_second_parabola.png"
)