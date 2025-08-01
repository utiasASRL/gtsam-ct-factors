### Guide to Creating a `MatrixLieGroup` Class in GTSAM

This guide outlines the minimal requirements for creating a new, fixed-size matrix Lie group (e.g., `MyGroup`) in GTSAM. The framework leverages the Curiously Recurring Template Pattern (CRTP), meaning that by inheriting from `gtsam::MatrixLieGroup`, your class automatically gets many of its required methods implemented for free. This guide distinguishes between what you **must** implement and what is **automatically provided**.

---

### 1. What You MUST Implement

These are the essential components you need to define in your class for it to be a fully functional, concept-compliant `MatrixLieGroup`.

#### `MyGroup.h` - Header File Requirements

1.  **Class Definition and Inheritance**:
    *   Inherit publicly from `gtsam::MatrixLieGroup<MyGroup, D, N>`, where `D` is the dimension of the tangent space and `N` is the side-length of the matrix.
    *   Contain a member variable for the `N x N` matrix, e.g., `gtsam::MatrixN T_;`.

2.  **Essential Typedefs and Constants**:
    *   `static const size_t dimension = D;`
    *   `using LieAlgebra = gtsam::MatrixN;`
    *   `using ChartJacobian = gtsam::OptionalJacobian<D, D>;`

3.  **Constructors**:
    *   A default constructor that initializes to the identity element.
    *   A constructor from an `N x N` matrix. This is where you should perform any necessary normalization (e.g., for `SL(n)`, ensure the determinant is 1).

4.  **Group Primitives**:
    *   `static MyGroup Identity();`
    *   `MyGroup inverse() const;`
    *   `MyGroup operator*(const MyGroup& other) const;`

5.  **Lie Group Primitives**:
    *   `static MyGroup Expmap(const gtsam::VectorD& xi, ChartJacobian H = {});`: The Exponential map. **Crucially, it must support an optional derivative argument to be concept-compliant.**
    *   `static gtsam::VectorD Logmap(const MyGroup& p, ChartJacobian H = {});`: The Logarithm map. **It must also support an optional derivative argument.**
    *   `gtsam::MatrixD AdjointMap() const;`: Computes the Adjoint map. While `MatrixLieGroup` provides a generic (but slow) default, a closed-form version is strongly recommended for performance.
    *   `ChartAtOrigin` struct: A first-order approximation for Expmap/Logmap.
        *   `static MyGroup Retract(const gtsam::VectorD& v, ChartJacobian H = {});`
        *   `static gtsam::VectorD Local(const MyGroup& p, ChartJacobian H = {});`

6.  **Matrix Lie Group Primitives**:
    *   `const gtsam::MatrixN& matrix() const;`: An accessor for the underlying matrix.
    *   `static gtsam::MatrixN Hat(const gtsam::VectorD& xi);`: Maps a tangent vector to a matrix in the Lie algebra.
    *   `static gtsam::VectorD Vee(const gtsam::MatrixN& X);`: The inverse of `Hat`.

7.  **`using` Declaration for `inverse`**:
    *   `using LieGroup<MyGroup, D>::inverse;`: To prevent the `inverse() const` method from hiding the `inverse(ChartJacobian H)` method provided by the base class.

8.  **Utilities (for `Testable` concept)**:
    *   `void print(const std::string& s) const;`
    *   `bool equals(const MyGroup& other, double tol) const;`

9.  **GTSAM Traits Specialization**:
    *   Outside the class, specialize the `traits` struct. This is essential for integrating the class with GTSAM's type system.
    ```cpp
    namespace gtsam {
    template <>
    struct traits<MyGroup> : public internal::MatrixLieGroup<MyGroup, N> {};

    template <>
    struct traits<const MyGroup> : public internal::MatrixLieGroup<MyGroup, N> {};
    }
    ```

#### `MyGroup.cpp` - Implementation File

The `.cpp` file must implement the methods declared in the header. For `Expmap` and `Logmap`, the implementation must check if the optional Jacobian `H` is requested and compute it if it is.

---

### 2. What Is Automatically Provided (Do NOT Implement)

By inheriting from `gtsam::MatrixLieGroup` and implementing the primitives above, you get the following methods for free. They are implemented in `gtsam::LieGroup` and `gtsam::MatrixLieGroup` using your primitives.

*   `compose(const MyGroup& other)`: Composes `*this` with `other`.
*   `between(const MyGroup& other)`: Calculates the relative transformation from `*this` to `other`.
*   `retract(const VectorD& v)`: Applies a tangent vector `v` to `*this`.
*   `localCoordinates(const MyGroup& other)`: Finds the tangent vector that maps `*this` to `other`.
*   `expmap(const VectorD& v)`: The non-static version of `Expmap`, which composes `*this` with `Expmap(v)`.
*   `logmap(const MyGroup& other)`: The non-static version of `Logmap`.
*   **All Jacobian versions of the above methods**: e.g., `compose(g, H1, H2)`, `between(g, H1, H2)`, `inverse(H)`, etc. The base class provides the correct derivative calculations based on your `AdjointMap` and static `Expmap`/`Logmap` Jacobians.

---

### 3. Common but Superfluous/Legacy Methods

You may see the following methods implemented in existing GTSAM classes like `SL4`. For a new, **fixed-size** Lie group, they are not necessary.

*   `static size_t Dim()`
    *   **Reasoning**: This is redundant. The static constant `MyGroup::dimension` is the modern way to access the dimension at compile time, and it's what the `traits` system uses.

*   `size_t dim() const`
    *   **Reasoning**: This instance method is required for **dynamically-sized** manifolds where the dimension must be retrieved from the object at runtime. For a fixed-size group like `SL4`, it just returns the compile-time constant `dimension` and is therefore superfluous. It's harmless but not part of a minimal implementation.

Including these methods does not break anything and may have been conventional in the past, but they are not required to fulfill the modern `MatrixLieGroup` concept for fixed-size types.