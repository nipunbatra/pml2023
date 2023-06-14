import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def visualize_matrix(a11, a12, a21, a22, v1, v2):
    A = np.array([[a11, a12], [a21, a22]])
    v = np.array([[v1], [v2]])
    w = A @ v

    fig, ax = plt.subplots()
    ax.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1, color="blue")
    ax.quiver(0, 0, w[0], w[1], angles="xy", scale_units="xy", scale=1, color="red")

    # Add legend
    ax.legend(["v", "w"])
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    st.pyplot(fig)

    st.write(f"Matrix A: \n{A}")
    st.write(f"Vector v: \n{v}")
    st.write(f"Result w: \n{w}")


left_column, right_column = st.columns(2)

with left_column:
    st.write("Modify the matrix A")
    a11 = st.slider("a11", -5.0, 5.0, 1.0)
    a12 = st.slider("a12", -5.0, 5.0, 2.0)
    a21 = st.slider("a21", -5.0, 5.0, -1.0)
    a22 = st.slider("a22", -5.0, 5.0, 2.0)

with right_column:
    st.write("Modify the vector v")
    v1 = st.slider("v1", -5.0, 5.0, 1.0)
    v2 = st.slider("v2", -5.0, 5.0, 2.0)

visualize_matrix(a11, a12, a21, a22, v1, v2)
