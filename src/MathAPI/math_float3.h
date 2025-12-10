// Description: 3-dimensional vector class with comprehensive 
//              mathematical operations and SSE optimization
// Author: NSDeathman, DeepSeek
#pragma once

#include <xmmintrin.h>
#include <pmmintrin.h> // SSE3
#include <string>
#include <cstdio>
#include <cmath>
#include <algorithm>

#include "math_config.h"
#include "math_constants.h"
#include "math_functions.h"
#include "math_float2.h"

// Platform-specific support
#if defined(MATH_SUPPORT_D3DX)
#include <d3dx9.h>  // D3DXVECTOR3, D3DXVECTOR4, D3DCOLOR
#endif

namespace Math
{
    /**
     * @class float3
     * @brief 3-dimensional vector with comprehensive mathematical operations
     *
     * Represents a 3D vector (x, y, z) with optimized operations for 3D graphics,
     * physics simulations, and 3D mathematics. Features SSE optimization for
     * performance-critical operations.
     *
     * @note Perfect for 3D game development, computer graphics, and physics engines
     * @note All operations are optimized and constexpr where possible
     * @note Includes comprehensive HLSL-like function set
     */
    class MATH_API float3
    {
    public:
        // ============================================================================
        // Data Members (Public for Direct Access)
        // ============================================================================

        union {
            struct {
                float x; ///< X component of the vector
                float y; ///< Y component of the vector  
                float z; ///< Z component of the vector
                float _w; ///< Padding component for SSE alignment
            };
            __m128 simd_; ///< SSE register for optimized operations
        };

        // ============================================================================
        // Constructors
        // ============================================================================

        /**
         * @brief Default constructor (initializes to zero vector)
         */
        float3() noexcept;

        /**
         * @brief Construct from components
         * @param x X component
         * @param y Y component
         * @param z Z component
         */
        float3(float x, float y, float z) noexcept;

        /**
         * @brief Construct from scalar (all components set to same value)
         * @param scalar Value for all components
         */
        explicit float3(float scalar) noexcept;

        /**
         * @brief Construct from float2 and z component
         * @param vec 2D vector for x and y components
         * @param z Z component
         */
        float3(const float2& vec, float z = 0.0f) noexcept;

        /**
         * @brief Copy constructor
         */
        float3(const float3&) noexcept = default;

        /**
         * @brief Construct from raw float array
         * @param data Pointer to float array [x, y, z]
         */
        explicit float3(const float* data) noexcept;

        /**
         * @brief Construct from SSE register (advanced users)
         * @param simd_val SSE register containing vector data
         */
        explicit float3(__m128 simd_val) noexcept;

#if defined(MATH_SUPPORT_D3DX)
        /**
         * @brief Construct from D3DXVECTOR3
         * @param vec DirectX 3D vector
         */
        float3(const D3DXVECTOR3& vec) noexcept;

        /**
         * @brief Construct from D3DXVECTOR4 (extracts x, y, z)
         * @param vec DirectX 4D vector
         */
        float3(const D3DXVECTOR4& vec) noexcept;

        /**
         * @brief Construct from D3DXVECTOR2 and z component
         * @param vec DirectX 2D vector
         * @param z Z component
         */
        float3(const D3DXVECTOR2& vec, float z = 0.0f) noexcept;

        /**
         * @brief Construct from D3DCOLOR (RGB components)
         * @param color DirectX color value
         */
        explicit float3(D3DCOLOR color) noexcept;
#endif

        // ============================================================================
        // Assignment Operators
        // ============================================================================

        /**
         * @brief Copy assignment operator
         */
        float3& operator=(const float3&) noexcept = default;

        /**
         * @brief Scalar assignment (sets all components to same value)
         * @param scalar Value for all components
         */
        float3& operator=(float scalar) noexcept;

#if defined(MATH_SUPPORT_D3DX)
        /**
         * @brief Assignment from D3DXVECTOR3
         * @param vec DirectX 3D vector
         */
        float3& operator=(const D3DXVECTOR3& vec) noexcept;

        /**
         * @brief Assignment from D3DCOLOR
         * @param color DirectX color value
         */
        float3& operator=(D3DCOLOR color) noexcept;
#endif

        // ============================================================================
        // Compound Assignment Operators
        // ============================================================================

        /**
         * @brief Component-wise addition assignment
         * @param rhs Right-hand side vector
         * @return Reference to this vector
         */
        float3& operator+=(const float3& rhs) noexcept;

        /**
         * @brief Component-wise subtraction assignment
         * @param rhs Right-hand side vector
         * @return Reference to this vector
         */
        float3& operator-=(const float3& rhs) noexcept;

        /**
         * @brief Component-wise multiplication assignment
         * @param rhs Right-hand side vector
         * @return Reference to this vector
         */
        float3& operator*=(const float3& rhs) noexcept;

        /**
         * @brief Component-wise division assignment
         * @param rhs Right-hand side vector
         * @return Reference to this vector
         */
        float3& operator/=(const float3& rhs) noexcept;

        /**
         * @brief Scalar multiplication assignment
         * @param scalar Scalar multiplier
         * @return Reference to this vector
         */
        float3& operator*=(float scalar) noexcept;

        /**
         * @brief Scalar division assignment
         * @param scalar Scalar divisor
         * @return Reference to this vector
         */
        float3& operator/=(float scalar) noexcept;

        // ============================================================================
        // Unary Operators
        // ============================================================================

        /**
         * @brief Unary plus operator
         * @return Copy of this vector
         */
        float3 operator+() const noexcept;

        /**
         * @brief Unary minus operator
         * @return Negated vector
         */
        float3 operator-() const noexcept;

        // ============================================================================
        // Access Operators
        // ============================================================================

        /**
         * @brief Access component by index
         * @param index Component index (0 = x, 1 = y, 2 = z)
         * @return Reference to component
         */
        float& operator[](int index) noexcept;

        /**
         * @brief Access component by index (const)
         * @param index Component index (0 = x, 1 = y, 2 = z)
         * @return Const reference to component
         */
        const float& operator[](int index) const noexcept;

        // ============================================================================
        // Conversion Operators
        // ============================================================================

        /**
         * @brief Convert to const float pointer (for interoperability)
         * @return Pointer to const float array [x, y, z]
         */
        operator const float* () const noexcept;

        /**
         * @brief Convert to float pointer (for interoperability)
         * @return Pointer to float array [x, y, z]
         */
        operator float* () noexcept;

        /**
         * @brief Convert to SSE register (advanced users)
         * @return SSE register containing vector data
         */
        operator __m128() const noexcept;

#if defined(MATH_SUPPORT_D3DX)
        /**
         * @brief Convert to D3DXVECTOR3
         * @return D3DXVECTOR3 equivalent
         */
        operator D3DXVECTOR3() const noexcept;
#endif

        // ============================================================================
        // Static Constructors
        // ============================================================================

        /**
         * @brief Zero vector (0, 0, 0)
         * @return Zero vector
         */
        static float3 zero() noexcept;

        /**
         * @brief One vector (1, 1, 1)
         * @return One vector
         */
        static float3 one() noexcept;

        /**
         * @brief Unit X vector (1, 0, 0)
         * @return Unit X vector
         */
        static float3 unit_x() noexcept;

        /**
         * @brief Unit Y vector (0, 1, 0)
         * @return Unit Y vector
         */
        static float3 unit_y() noexcept;

        /**
         * @brief Unit Z vector (0, 0, 1)
         * @return Unit Z vector
         */
        static float3 unit_z() noexcept;

        /**
         * @brief Forward vector (0, 0, 1) - common in 3D graphics
         * @return Forward vector
         */
        static float3 forward() noexcept;

        /**
         * @brief Up vector (0, 1, 0) - common in 3D graphics
         * @return Up vector
         */
        static float3 up() noexcept;

        /**
         * @brief Right vector (1, 0, 0) - common in 3D graphics
         * @return Right vector
         */
        static float3 right() noexcept;

        // ============================================================================
        // Mathematical Functions
        // ============================================================================

        /**
         * @brief Compute Euclidean length (magnitude)
         * @return Length of the vector
         */
        float length() const noexcept;

        /**
         * @brief Compute squared length (faster, useful for comparisons)
         * @return Squared length of the vector
         */
        float length_sq() const noexcept;

        /**
         * @brief Normalize vector to unit length
         * @return Normalized vector
         * @note Returns zero vector if length is zero
         */
        float3 normalize() const noexcept;

        /**
         * @brief Compute dot product with another vector
         * @param other Other vector
         * @return Dot product result
         */
        float dot(const float3& other) const noexcept;

        /**
         * @brief Compute cross product with another vector
         * @param other Other vector
         * @return Cross product result
         */
        float3 cross(const float3& other) const noexcept;

        /**
         * @brief Compute distance to another point
         * @param other Other point
         * @return Euclidean distance
         */
        float distance(const float3& other) const noexcept;

        /**
         * @brief Compute squared distance to another point (faster)
         * @param other Other point
         * @return Squared Euclidean distance
         */
        float distance_sq(const float3& other) const noexcept;

        // ============================================================================
        // HLSL-like Functions
        // ============================================================================

        /**
         * @brief HLSL-like abs function (component-wise absolute value)
         * @return Vector with absolute values of components
         */
        float3 abs() const noexcept;

        /**
         * @brief HLSL-like sign function (component-wise sign)
         * @return Vector with signs of components (-1, 0, or 1)
         */
        float3 sign() const noexcept;

        /**
         * @brief HLSL-like floor function (component-wise floor)
         * @return Vector with floored components
         */
        float3 floor() const noexcept;

        /**
         * @brief HLSL-like ceil function (component-wise ceiling)
         * @return Vector with ceiling components
         */
        float3 ceil() const noexcept;

        /**
         * @brief HLSL-like round function (component-wise rounding)
         * @return Vector with rounded components
         */
        float3 round() const noexcept;

        /**
         * @brief HLSL-like frac function (component-wise fractional part)
         * @return Vector with fractional parts of components
         */
        float3 frac() const noexcept;

        /**
         * @brief HLSL-like saturate function (clamp components to [0, 1])
         * @return Saturated vector
         */
        float3 saturate() const noexcept;

        /**
         * @brief HLSL-like step function (component-wise step)
         * @param edge Edge value
         * @return 1.0 if component >= edge, else 0.0
         */
        float3 step(float edge) const noexcept;

        // ============================================================================
        // Geometric Operations
        // ============================================================================

        /**
         * @brief Compute reflection vector
         * @param normal Surface normal (must be normalized)
         * @return Reflected vector
         */
        float3 reflect(const float3& normal) const noexcept;

        /**
         * @brief Compute refraction vector
         * @param normal Surface normal (must be normalized)
         * @param eta Ratio of indices of refraction
         * @return Refracted vector
         */
        float3 refract(const float3& normal, float eta) const noexcept;

        /**
         * @brief Project vector onto another vector
         * @param onto Vector to project onto
         * @return Projected vector
         */
        float3 project(const float3& onto) const noexcept;

        /**
         * @brief Reject vector from another vector (component perpendicular)
         * @param from Vector to reject from
         * @return Rejected vector
         */
        float3 reject(const float3& from) const noexcept;

        // ============================================================================
        // Static Mathematical Functions
        // ============================================================================

        /**
         * @brief Compute dot product of two vectors
         * @param a First vector
         * @param b Second vector
         * @return Dot product result
         */
        static float dot(const float3& a, const float3& b) noexcept;

        /**
         * @brief Compute cross product of two vectors
         * @param a First vector
         * @param b Second vector
         * @return Cross product result
         */
        static float3 cross(const float3& a, const float3& b) noexcept;

        /**
         * @brief Linear interpolation between two vectors
         * @param a Start vector
         * @param b End vector
         * @param t Interpolation factor [0, 1]
         * @return Interpolated vector
         */
        static float3 lerp(const float3& a, const float3& b, float t) noexcept;

        /**
         * @brief Spherical linear interpolation (for directions)
         * @param a Start vector (should be normalized)
         * @param b End vector (should be normalized)
         * @param t Interpolation factor [0, 1]
         * @return Interpolated vector
         */
        static float3 slerp(const float3& a, const float3& b, float t) noexcept;

        /**
         * @brief Component-wise minimum of two vectors
         * @param a First vector
         * @param b Second vector
         * @return Component-wise minimum
         */
        static float3 min(const float3& a, const float3& b) noexcept;

        /**
         * @brief Component-wise maximum of two vectors
         * @param a First vector
         * @param b Second vector
         * @return Component-wise maximum
         */
        static float3 max(const float3& a, const float3& b) noexcept;

        /**
         * @brief HLSL-like saturate function (clamp components to [0, 1])
         * @param vec Vector to saturate
         * @return Saturated vector
         */
        static float3 saturate(const float3& vec) noexcept;

        /**
         * @brief Compute reflection vector
         * @param incident Incident vector
         * @param normal Surface normal (must be normalized)
         * @return Reflected vector
         */
        static float3 reflect(const float3& incident, const float3& normal) noexcept;

        /**
         * @brief Compute refraction vector
         * @param incident Incident vector
         * @param normal Surface normal (must be normalized)
         * @param eta Ratio of indices of refraction
         * @return Refracted vector
         */
        static float3 refract(const float3& incident, const float3& normal, float eta) noexcept;

        // ============================================================================
        // Swizzle Operations (HLSL style)
        // ============================================================================

        /**
         * @brief Get XY components as float2
         * @return 2D vector with x and y components
         */
        float2 xy() const noexcept;

        /**
         * @brief Get XZ components as float2
         * @return 2D vector with x and z components
         */
        float2 xz() const noexcept;

        /**
         * @brief Get YZ components as float2
         * @return 2D vector with y and z components
         */
        float2 yz() const noexcept;

        /**
         * @brief Get YX components as float2 (swapped)
         * @return 2D vector with y and x components
         */
        float2 yx() const noexcept;

        /**
         * @brief Get ZX components as float2
         * @return 2D vector with z and x components
         */
        float2 zx() const noexcept;

        /**
         * @brief Get ZY components as float2
         * @return 2D vector with z and y components
         */
        float2 zy() const noexcept;

        /**
         * @brief Get YXZ swizzle as float3
         * @return 3D vector with components reordered as (y, x, z)
         */
        float3 yxz() const noexcept;

        /**
         * @brief Get ZXY swizzle as float3
         * @return 3D vector with components reordered as (z, x, y)
         */
        float3 zxy() const noexcept;

        /**
         * @brief Get ZYX swizzle as float3
         * @return 3D vector with components reordered as (z, y, x)
         */
        float3 zyx() const noexcept;

        /**
         * @brief Get XZY swizzle as float3
         * @return 3D vector with components reordered as (x, z, y)
         */
        float3 xzy() const noexcept;

        /**
         * @brief Get XYX swizzle as float3
         * @return 3D vector with components reordered as (x, y, x)
         */
        float3 xyx() const noexcept;

        /**
         * @brief Get XYZ swizzle as float3
         * @return 3D vector with components reordered as (x, y, z)
         */
        float3 xyz() const noexcept;

        /**
         * @brief Get XZX swizzle as float3
         * @return 3D vector with components reordered as (x, z, x)
         */
        float3 xzx() const noexcept;

        /**
         * @brief Get YXY swizzle as float3
         * @return 3D vector with components reordered as (y, x, y)
         */
        float3 yxy() const noexcept;

        /**
         * @brief Get YZY swizzle as float3
         * @return 3D vector with components reordered as (y, z, y)
         */
        float3 yzy() const noexcept;

        /**
         * @brief Get ZXZ swizzle as float3
         * @return 3D vector with components reordered as (z, x, z)
         */
        float3 zxz() const noexcept;

        /**
         * @brief Get ZYZ swizzle as float3
         * @return 3D vector with components reordered as (z, y, z)
         */
        float3 zyz() const noexcept;

        // Color swizzles
        /**
         * @brief Get red component (alias for x)
         * @return Red component
         */
        float r() const noexcept;

        /**
         * @brief Get green component (alias for y)
         * @return Green component
         */
        float g() const noexcept;

        /**
         * @brief Get blue component (alias for z)
         * @return Blue component
         */
        float b() const noexcept;

        /**
         * @brief Get RG components as float2
         * @return 2D vector with red and green components
         */
        float2 rg() const noexcept;

        /**
         * @brief Get RB components as float2
         * @return 2D vector with red and blue components
         */
        float2 rb() const noexcept;

        /**
         * @brief Get GB components as float2
         * @return 2D vector with green and blue components
         */
        float2 gb() const noexcept;

        /**
         * @brief Get RGB components as float3
         * @return 3D vector with red, green, and blue components
         */
        float3 rgb() const noexcept;

        /**
         * @brief Get BGR components as float3
         * @return 3D vector with blue, green, and red components
         */
        float3 bgr() const noexcept;

        /**
         * @brief Get GBR components as float3
         * @return 3D vector with green, blue, and red components
         */
        float3 gbr() const noexcept;

        // ============================================================================
        // Utility Methods
        // ============================================================================

        /**
         * @brief Check if vector contains finite values
         * @return True if all components are finite (not NaN or infinity)
         */
        bool isValid() const noexcept;

        /**
         * @brief Check if vector is approximately equal to another
         * @param other Vector to compare with
         * @param epsilon Comparison tolerance
         * @return True if vectors are approximately equal
         */
        bool approximately(const float3& other, float epsilon = EPSILON) const noexcept;

        /**
         * @brief Check if vector is approximately zero
         * @param epsilon Comparison tolerance
         * @return True if vector length is approximately zero
         */
        bool approximately_zero(float epsilon = EPSILON) const noexcept;

        /**
         * @brief Check if vector is normalized
         * @param epsilon Comparison tolerance
         * @return True if vector length is approximately 1.0
         */
        bool is_normalized(float epsilon = 0.001f) const noexcept;

        /**
         * @brief Convert to string representation
         * @return String in format "(x, y, z)"
         */
        std::string to_string() const;

        /**
         * @brief Get pointer to raw data
         * @return Pointer to first component
         */
        const float* data() const noexcept;

        /**
         * @brief Get pointer to raw data (mutable)
         * @return Pointer to first component
         */
        float* data() noexcept;

        /**
         * @brief Set x and y components from float2
         * @param xy 2D vector for x and y components
         */
        void set_xy(const float2& xy) noexcept;

        /**
         * @brief Get SSE register (advanced users)
         * @return SSE register containing vector data
         */
        __m128 get_simd() const noexcept;

        /**
         * @brief Set SSE register (advanced users)
         * @param new_simd SSE register to set
         */
        void set_simd(__m128 new_simd) noexcept;

        // ============================================================================
        // Comparison Operators
        // ============================================================================

        /**
         * @brief Equality comparison
         * @param rhs Right-hand side vector
         * @return True if vectors are approximately equal
         */
        bool operator==(const float3& rhs) const noexcept;

        /**
         * @brief Inequality comparison
         * @param rhs Right-hand side vector
         * @return True if vectors are not approximately equal
         */
        bool operator!=(const float3& rhs) const noexcept;
    };

    // ============================================================================
    // Binary Operators
    // ============================================================================

    /**
     * @brief Vector addition
     * @param lhs Left-hand side vector
     * @param rhs Right-hand side vector
     * @return Result of addition
     */
    inline float3 operator+(float3 lhs, const float3& rhs) noexcept;

    /**
     * @brief Vector subtraction
     * @param lhs Left-hand side vector
     * @param rhs Right-hand side vector
     * @return Result of subtraction
     */
    inline float3 operator-(float3 lhs, const float3& rhs) noexcept;

    /**
     * @brief Component-wise vector multiplication
     * @param lhs Left-hand side vector
     * @param rhs Right-hand side vector
     * @return Result of multiplication
     */
    inline float3 operator*(float3 lhs, const float3& rhs) noexcept;

    /**
     * @brief Component-wise vector division
     * @param lhs Left-hand side vector
     * @param rhs Right-hand side vector
     * @return Result of division
     */
    inline float3 operator/(float3 lhs, const float3& rhs) noexcept;

    /**
     * @brief Vector-scalar multiplication
     * @param vec Vector to multiply
     * @param scalar Scalar multiplier
     * @return Scaled vector
     */
    inline float3 operator*(float3 vec, float scalar) noexcept;

    /**
     * @brief Scalar-vector multiplication
     * @param scalar Scalar multiplier
     * @param vec Vector to multiply
     * @return Scaled vector
     */
    inline float3 operator*(float scalar, float3 vec) noexcept;

    /**
     * @brief Vector-scalar division
     * @param vec Vector to divide
     * @param scalar Scalar divisor
     * @return Scaled vector
     */
    inline float3 operator/(float3 vec, float scalar) noexcept;

    // ============================================================================
    // Global Mathematical Functions
    // ============================================================================

    /**
     * @brief Compute distance between two points
     * @param a First point
     * @param b Second point
     * @return Euclidean distance between points
     */
    inline float distance(const float3& a, const float3& b) noexcept;

    /**
     * @brief Compute squared distance between two points (faster)
     * @param a First point
     * @param b Second point
     * @return Squared Euclidean distance
     */
    inline float distance_sq(const float3& a, const float3& b) noexcept;

    /**
     * @brief Compute dot product of two vectors
     * @param a First vector
     * @param b Second vector
     * @return Dot product result
     */
    inline float dot(const float3& a, const float3& b) noexcept;

    /**
     * @brief Compute cross product of two vectors
     * @param a First vector
     * @param b Second vector
     * @return Cross product result
     */
    inline float3 cross(const float3& a, const float3& b) noexcept;

    /**
     * @brief Normalize vector to unit length
     * @param vec Vector to normalize
     * @return Normalized vector
     */
    inline float3 normalize(const float3& vec) noexcept;

    /**
     * @brief Linear interpolation between two vectors
     * @param a Start vector
     * @param b End vector
     * @param t Interpolation factor [0, 1]
     * @return Interpolated vector
     */
    inline float3 lerp(const float3& a, const float3& b, float t) noexcept;

    /**
     * @brief Spherical linear interpolation between two vectors
     * @param a Start vector (should be normalized)
     * @param b End vector (should be normalized)
     * @param t Interpolation factor [0, 1]
     * @return Interpolated vector
     */
    inline float3 slerp(const float3& a, const float3& b, float t) noexcept;

    /**
     * @brief Check if two vectors are approximately equal
     * @param a First vector
     * @param b Second vector
     * @param epsilon Comparison tolerance
     * @return True if vectors are approximately equal
     */
    inline bool approximately(const float3& a, const float3& b, float epsilon = EPSILON) noexcept;

    /**
     * @brief Check if vector is normalized
     * @param vec Vector to check
     * @param epsilon Comparison tolerance
     * @return True if vector length is approximately 1.0
     */
    inline bool is_normalized(const float3& vec, float epsilon = EPSILON) noexcept;

    /**
     * @brief Check if two vectors are orthogonal (perpendicular)
     * @param a First vector
     * @param b Second vector
     * @param epsilon Comparison tolerance
     * @return True if dot product is approximately zero
     */
    inline bool are_orthogonal(const float3& a, const float3& b, float epsilon = EPSILON) noexcept;

    /**
     * @brief Check if three vectors form an orthonormal basis
     * @param x X basis vector
     * @param y Y basis vector
     * @param z Z basis vector
     * @param epsilon Comparison tolerance
     * @return True if all vectors are normalized and mutually orthogonal
     */
    inline bool is_orthonormal_basis(const float3& x, const float3& y, const float3& z, float epsilon = EPSILON) noexcept;

    /**
     * @brief Check if vector contains valid finite values
     * @param vec Vector to check
     * @return True if vector is valid (finite values)
     */
    inline bool isValid(const float3& vec) noexcept;

    // ============================================================================
    // HLSL-like Global Functions
    // ============================================================================

    /**
     * @brief HLSL-like abs function (component-wise absolute value)
     * @param vec Input vector
     * @return Vector with absolute values of components
     */
    inline float3 abs(const float3& vec) noexcept;

    /**
     * @brief HLSL-like sign function (component-wise sign)
     * @param vec Input vector
     * @return Vector with signs of components
     */
    inline float3 sign(const float3& vec) noexcept;

    /**
     * @brief HLSL-like floor function (component-wise floor)
     * @param vec Input vector
     * @return Vector with floored components
     */
    inline float3 floor(const float3& vec) noexcept;

    /**
     * @brief HLSL-like ceil function (component-wise ceiling)
     * @param vec Input vector
     * @return Vector with ceiling components
     */
    inline float3 ceil(const float3& vec) noexcept;

    /**
     * @brief HLSL-like round function (component-wise rounding)
     * @param vec Input vector
     * @return Vector with rounded components
     */
    inline float3 round(const float3& vec) noexcept;

    /**
     * @brief HLSL-like frac function (component-wise fractional part)
     * @param vec Input vector
     * @return Vector with fractional parts of components
     */
    inline float3 frac(const float3& vec) noexcept;

    /**
     * @brief HLSL-like saturate function (clamp components to [0, 1])
     * @param vec Input vector
     * @return Saturated vector
     */
    inline float3 saturate(const float3& vec) noexcept;

    /**
     * @brief HLSL-like step function (component-wise step)
     * @param edge Edge value
     * @param vec Input vector
     * @return Step result vector
     */
    inline float3 step(float edge, const float3& vec) noexcept;

    /**
     * @brief HLSL-like min function (component-wise minimum)
     * @param a First vector
     * @param b Second vector
     * @return Component-wise minimum
     */
    inline float3 min(const float3& a, const float3& b) noexcept;

    /**
     * @brief HLSL-like max function (component-wise maximum)
     * @param a First vector
     * @param b Second vector
     * @return Component-wise maximum
     */
    inline float3 max(const float3& a, const float3& b) noexcept;

    /**
     * @brief HLSL-like clamp function (component-wise clamping)
     * @param vec Vector to clamp
     * @param min_val Minimum values
     * @param max_val Maximum values
     * @return Clamped vector
     */
    inline float3 clamp(const float3& vec, const float3& min_val, const float3& max_val) noexcept;

    /**
     * @brief Scalar clamp function (all components clamped to same range)
     * @param vec Vector to clamp
     * @param min_val Minimum value for all components
     * @param max_val Maximum value for all components
     * @return Clamped vector
     */
    inline float3 clamp(const float3& vec, float min_val, float max_val) noexcept;

    /**
     * @brief Compute vector length
     * @param vec Input vector
     * @return Length of the vector
     */
    inline float length(const float3& vec) noexcept;

    /**
     * @brief Compute squared vector length
     * @param vec Input vector
     * @return Squared length of the vector
     */
    inline float length_sq(const float3& vec) noexcept;

    // ============================================================================
    // Geometric Operations
    // ============================================================================

    /**
     * @brief Compute reflection vector
     * @param incident Incident vector
     * @param normal Surface normal (must be normalized)
     * @return Reflected vector
     */
    inline float3 reflect(const float3& incident, const float3& normal) noexcept;

    /**
     * @brief Compute refraction vector
     * @param incident Incident vector
     * @param normal Surface normal (must be normalized)
     * @param eta Ratio of indices of refraction
     * @return Refracted vector
     */
    inline float3 refract(const float3& incident, const float3& normal, float eta) noexcept;

    /**
     * @brief Project vector onto another vector
     * @param vec Vector to project
     * @param onto Vector to project onto
     * @return Projected vector
     */
    inline float3 project(const float3& vec, const float3& onto) noexcept;

    /**
     * @brief Reject vector from another vector (component perpendicular)
     * @param vec Vector to reject
     * @param from Vector to reject from
     * @return Rejected vector
     */
    inline float3 reject(const float3& vec, const float3& from) noexcept;

    /**
     * @brief Compute angle between two vectors in radians
     * @param a First vector
     * @param b Second vector
     * @return Angle in radians between [0, π]
     */
    inline float angle_between(const float3& a, const float3& b) noexcept;

    // ============================================================================
    // D3D Compatibility Functions
    // ============================================================================

#if defined(MATH_SUPPORT_D3DX)

    /**
     * @brief Convert float3 to D3DXVECTOR3
     * @param vec float3 vector to convert
     * @return D3DXVECTOR3 equivalent
     */
    inline D3DXVECTOR3 ToD3DXVECTOR3(const float3& vec) noexcept;

    /**
     * @brief Convert D3DXVECTOR3 to float3
     * @param vec D3DXVECTOR3 to convert
     * @return float3 equivalent
     */
    inline float3 FromD3DXVECTOR3(const D3DXVECTOR3& vec) noexcept;

    /**
     * @brief Convert float3 to D3DCOLOR (uses x,y,z as R,G,B channels)
     * @param color float3 representing color (x=R, y=G, z=B)
     * @return D3DCOLOR equivalent
     */
    inline D3DCOLOR ToD3DCOLOR(const float3& color) noexcept;

    /**
     * @brief Convert array of float3 to array of D3DXVECTOR3
     * @param source Source float3 array
     * @param destination Destination D3DXVECTOR3 array
     * @param count Number of elements to convert
     */
    inline void float3ArrayToD3D(const float3* source, D3DXVECTOR3* destination, size_t count) noexcept;

    /**
     * @brief Convert array of D3DXVECTOR3 to array of float3
     * @param source Source D3DXVECTOR3 array
     * @param destination Destination float3 array
     * @param count Number of elements to convert
     */
    inline void D3DArrayTofloat3(const D3DXVECTOR3* source, float3* destination, size_t count) noexcept;

#endif

    // ============================================================================
    // Useful Constants
    // ============================================================================

    /**
     * @brief Zero vector constant (0, 0, 0)
     */
    extern const float3 float3_Zero;

    /**
     * @brief One vector constant (1, 1, 1)
     */
    extern const float3 float3_One;

    /**
     * @brief Unit X vector constant (1, 0, 0)
     */
    extern const float3 float3_UnitX;

    /**
     * @brief Unit Y vector constant (0, 1, 0)
     */
    extern const float3 float3_UnitY;

    /**
     * @brief Unit Z vector constant (0, 0, 1)
     */
    extern const float3 float3_UnitZ;

    /**
     * @brief Forward vector constant (0, 0, 1)
     */
    extern const float3 float3_Forward;

    /**
     * @brief Up vector constant (0, 1, 0)
     */
    extern const float3 float3_Up;

    /**
     * @brief Right vector constant (1, 0, 0)
     */
    extern const float3 float3_Right;

} // namespace Math

// Include Implementation at the end
#include "math_float3.inl"
