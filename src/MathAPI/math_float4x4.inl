#ifndef MATH_FLOAT4X4_INL
#define MATH_FLOAT4X4_INL

#include "math_float4x4.h"
#include "math_float3x3.h"
#include "math_quaternion.h"

namespace Math
{
    // ============================================================================
    // Constructors
    // ============================================================================

    /**
     * @brief Default constructor (initializes to zero matrix)
     */
    inline float4x4::float4x4() noexcept
        : col0_(1, 0, 0, 0)
        , col1_(0, 1, 0, 0)
        , col2_(0, 0, 1, 0)
        , col3_(0, 0, 0, 1) {}

    /**
     * @brief Construct from column vectors
     * @param col0 First column vector
     * @param col1 Second column vector
     * @param col2 Third column vector
     * @param col3 Fourth column vector (translation/perspective)
     */
    inline float4x4::float4x4(const float4& col0, const float4& col1, const float4& col2, const float4& col3) noexcept
        : col0_(col0), col1_(col1), col2_(col2), col3_(col3) {}

    /**
     * @brief Construct from column-major array
     * @param data Column-major array of 16 elements
     * @note Expected order: [col0.x, col1.x, col2.x, col3.x, col0.y, ...]
     */
    inline float4x4::float4x4(const float* data) noexcept
        : col0_(data[0], data[4], data[8], data[12])
        , col1_(data[1], data[5], data[9], data[13])
        , col2_(data[2], data[6], data[10], data[14])
        , col3_(data[3], data[7], data[11], data[15]) {}

    inline float4x4::float4x4(float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33) noexcept
    {
        // Параметры в row-major порядке, сохраняем в column-major
        // row0: m00, m01, m02, m03 -> становится col0.x, col1.x, col2.x, col3.x
        // row1: m10, m11, m12, m13 -> становится col0.y, col1.y, col2.y, col3.y
        // row2: m20, m21, m22, m23 -> становится col0.z, col1.z, col2.z, col3.z  
        // row3: m30, m31, m32, m33 -> становится col0.w, col1.w, col2.w, col3.w

        col0_ = float4(m00, m10, m20, m30);
        col1_ = float4(m01, m11, m21, m31);
        col2_ = float4(m02, m12, m22, m32);
        col3_ = float4(m03, m13, m23, m33);
    }

    /**
     * @brief Construct from scalar (diagonal matrix)
     * @param scalar Value for diagonal elements
     */
    inline float4x4::float4x4(float scalar) noexcept
        : col0_(scalar, 0, 0, 0)
        , col1_(0, scalar, 0, 0)
        , col2_(0, 0, scalar, 0)
        , col3_(0, 0, 0, scalar) {}

    /**
     * @brief Construct from diagonal vector
     * @param diagonal Diagonal elements (x, y, z, w)
     */
    inline float4x4::float4x4(const float4& diagonal) noexcept
        : col0_(diagonal.x, 0, 0, 0)
        , col1_(0, diagonal.y, 0, 0)
        , col2_(0, 0, diagonal.z, 0)
        , col3_(0, 0, 0, diagonal.w) {}

    /**
     * @brief Construct from 3x3 matrix (extends to 4x4 with identity)
     * @param mat3x3 3x3 matrix to extend
     * @note Adds identity for translation and perspective components
     */
    inline float4x4::float4x4(const float3x3& mat3x3) noexcept
    {
        // Extract columns from 3x3 matrix (assuming float3x3 stores in column-major)
        const float3 col0 = mat3x3.col0();
        const float3 col1 = mat3x3.col1();
        const float3 col2 = mat3x3.col2();

        // Convert to column-major 4x4
        col0_ = float4(col0.x, col0.y, col0.z, 0.0f);
        col1_ = float4(col1.x, col1.y, col1.z, 0.0f);
        col2_ = float4(col2.x, col2.y, col2.z, 0.0f);
        col3_ = float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    /**
     * @brief Construct from quaternion (rotation matrix)
     * @param q Unit quaternion representing rotation
     * @note Creates homogeneous rotation matrix with no translation
     * @note Uses SSE optimization for efficient conversion
     */
    inline float4x4::float4x4(const quaternion& q) noexcept
    {
        // Normalize quaternion using SSE
        const __m128 q_simd = q.normalize().get_simd();

        // Extract components
        alignas(16) float q_data[4];
        _mm_store_ps(q_data, q_simd);
        const float x = q_data[0], y = q_data[1], z = q_data[2], w = q_data[3];

        // Compute common terms
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, xz = x * z, yz = y * z;
        const float wx = w * x, wy = w * y, wz = w * z;

        const float _2yy = 2.0f * yy, _2zz = 2.0f * zz;
        const float _2xy = 2.0f * xy, _2xz = 2.0f * xz, _2yz = 2.0f * yz;
        const float _2wx = 2.0f * wx, _2wy = 2.0f * wy, _2wz = 2.0f * wz;

        // Build matrix columns using SSE
        col0_ = float4(1.0f - _2yy - _2zz, _2xy + _2wz, _2xz - _2wy, 0.0f);
        col1_ = float4(_2xy - _2wz, 1.0f - 2.0f * (xx + zz), _2yz + _2wx, 0.0f);
        col2_ = float4(_2xz + _2wy, _2yz - _2wx, 1.0f - 2.0f * (xx + yy), 0.0f);
        col3_ = float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

#if defined(MATH_SUPPORT_D3DX)
    /**
     * @brief Construct from D3DXMATRIX
     * @param mat DirectX matrix
     * @note Converts from DirectX row-major to internal column-major storage
     */
    inline float4x4::float4x4(const D3DXMATRIX& mat) noexcept
        : col0_(mat._11, mat._21, mat._31, mat._41)
        , col1_(mat._12, mat._22, mat._32, mat._42)
        , col2_(mat._13, mat._23, mat._33, mat._43)
        , col3_(mat._14, mat._24, mat._34, mat._44) {}
#endif

    // ============================================================================
    // Assignment Operators
    // ============================================================================

#if defined(MATH_SUPPORT_D3DX)
    /**
     * @brief Assignment from D3DXMATRIX
     * @param mat DirectX matrix
     * @return Reference to this matrix
     * @note Converts from DirectX row-major to internal column-major storage
     */
    inline float4x4& float4x4::operator=(const D3DXMATRIX& mat) noexcept
    {
        col0_ = float4(mat._11, mat._21, mat._31, mat._41);
        col1_ = float4(mat._12, mat._22, mat._32, mat._42);
        col2_ = float4(mat._13, mat._23, mat._33, mat._43);
        col3_ = float4(mat._14, mat._24, mat._34, mat._44);
        return *this;
    }
#endif

    // ============================================================================
    // Static Constructors
    // ============================================================================

    /**
     * @brief Identity matrix
     * @return 4x4 identity matrix
     */
    inline float4x4 float4x4::identity() noexcept
    {
        return float4x4(
            float4(1, 0, 0, 0),
            float4(0, 1, 0, 0),
            float4(0, 0, 1, 0),
            float4(0, 0, 0, 1)
        );
    }

    /**
     * @brief Zero matrix
     * @return 4x4 zero matrix
     */
    inline float4x4 float4x4::zero() noexcept
    {
        return float4x4(
            float4(0, 0, 0, 0),
            float4(0, 0, 0, 0),
            float4(0, 0, 0, 0),
            float4(0, 0, 0, 0)
        );
    }

    /**
     * @brief Translation matrix
     * @param translation Translation vector
     * @return Translation matrix
     * @note Creates matrix that translates points by the specified vector
     */
    inline float4x4 float4x4::translation(const float3& translation) noexcept
    {
        return float4x4(
            float4(1.0f, 0.0f, 0.0f, 0.0f),
            float4(0.0f, 1.0f, 0.0f, 0.0f),
            float4(0.0f, 0.0f, 1.0f, 0.0f),
            float4(translation.x, translation.y, translation.z, 1.0f)
        );
    }

    /**
     * @brief Translation matrix from components
     * @param x X translation
     * @param y Y translation
     * @param z Z translation
     * @return Translation matrix
     */
    inline float4x4 float4x4::translation(float x, float y, float z) noexcept
    {
        return translation(float3(x, y, z));
    }

    /**
     * @brief Scaling matrix
     * @param scale Scale factors
     * @return Scaling matrix
     * @note Creates matrix that scales points by the specified factors
     */
    inline float4x4 float4x4::scaling(const float3& scale) noexcept
    {
        return float4x4(
            float4(scale.x, 0, 0, 0),
            float4(0, scale.y, 0, 0),
            float4(0, 0, scale.z, 0),
            float4(0, 0, 0, 1)
        );
    }

    /**
     * @brief Scaling matrix from components
     * @param x X scale
     * @param y Y scale
     * @param z Z scale
     * @return Scaling matrix
     */
    inline float4x4 float4x4::scaling(float x, float y, float z) noexcept
    {
        return scaling(float3(x, y, z));
    }

    /**
     * @brief Uniform scaling matrix
     * @param uniformScale Uniform scale factor
     * @return Scaling matrix
     */
    inline float4x4 float4x4::scaling(float uniformScale) noexcept
    {
        return scaling(float3(uniformScale));
    }

    /**
     * @brief Rotation matrix around X axis
     * @param angle Rotation angle in radians
     * @return Rotation matrix around X
     */
    inline float4x4 float4x4::rotation_x(float angle) noexcept
    {
        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);

        return float4x4(
            float4(1.0f, 0.0f, 0.0f, 0.0f),
            float4(0.0f, c, s, 0.0f),
            float4(0.0f, -s, c, 0.0f),
            float4(0.0f, 0.0f, 0.0f, 1.0f)
        );
    }

    /**
     * @brief Rotation matrix around Y axis
     * @param angle Rotation angle in radians
     * @return Rotation matrix around Y
     */
    inline float4x4 float4x4::rotation_y(float angle) noexcept
    {
        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);

        return float4x4(
            float4(c, 0, -s, 0),
            float4(0, 1, 0, 0),
            float4(s, 0, c, 0),
            float4(0, 0, 0, 1)
        );
    }

    /**
     * @brief Rotation matrix around Z axis
     * @param angle Rotation angle in radians
     * @return Rotation matrix around Z
     */
    inline float4x4 float4x4::rotation_z(float angle) noexcept
    {
        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);

        return float4x4(
            float4(c, s, 0, 0),
            float4(-s, c, 0, 0),
            float4(0, 0, 1, 0),
            float4(0, 0, 0, 1)
        );
    }

    /**
     * @brief Rotation matrix from axis and angle
     * @param axis Rotation axis (must be normalized)
     * @param angle Rotation angle in radians
     * @return Rotation matrix
     * @note Uses Rodrigues' rotation formula
     */
    inline float4x4 float4x4::rotation_axis(const float3& axis, float angle) noexcept
    {
        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);

        const float one_minus_c = 1.0f - c;
        const float xy = axis.x * axis.y;
        const float xz = axis.x * axis.z;
        const float yz = axis.y * axis.z;
        const float xs = axis.x * s;
        const float ys = axis.y * s;
        const float zs = axis.z * s;

        return float4x4(
            float4(one_minus_c * axis.x * axis.x + c,
                one_minus_c * xy + zs,
                one_minus_c * xz - ys, 0),
            float4(one_minus_c * xy - zs,
                one_minus_c * axis.y * axis.y + c,
                one_minus_c * yz + xs, 0),
            float4(one_minus_c * xz + ys,
                one_minus_c * yz - xs,
                one_minus_c * axis.z * axis.z + c, 0),
            float4(0, 0, 0, 1)
        );
    }

    /**
     * @brief Rotation matrix from Euler angles (ZYX order)
     * @param angles Euler angles in radians (pitch, yaw, roll)
     * @return Rotation matrix
     * @note Order: rotation_z(angles.z) * rotation_y(angles.y) * rotation_x(angles.x)
     * @note Pitch (X), Yaw (Y), Roll (Z) convention
     */
    inline float4x4 float4x4::rotation_euler(const float3& angles) noexcept
    {
        float sx, cx, sy, cy, sz, cz;
        MathFunctions::sin_cos(angles.x, &sx, &cx);
        MathFunctions::sin_cos(angles.y, &sy, &cy);
        MathFunctions::sin_cos(angles.z, &sz, &cz);

        // Прямое вычисление матрицы (Z * Y * X)
        return float4x4(
            float4(cy * cz, cx * sz + sx * sy * cz, sx * sz - cx * sy * cz, 0),
            float4(-cy * sz, cx * cz - sx * sy * sz, sx * cz + cx * sy * sz, 0),
            float4(sy, -sx * cy, cx * cy, 0),
            float4(0, 0, 0, 1)
        );
    }

    /**
     * @brief Orthographic projection matrix
     * @param left Left clipping plane
     * @param right Right clipping plane
     * @param bottom Bottom clipping plane
     * @param top Top clipping plane
     * @param zNear Near clipping plane
     * @param zFar Far clipping plane
     * @return Orthographic projection matrix
     * @note Creates parallel projection matrix
     */
    inline float4x4 float4x4::orthographic(float left, float right, float bottom, float top, float zNear, float zFar) noexcept
    {
        const float rcpWidth = 1.0f / (right - left);
        const float rcpHeight = 1.0f / (top - bottom);
        const float rcpDepth = 1.0f / (zFar - zNear);

        return float4x4(
            float4(2.0f * rcpWidth, 0.0f, 0.0f, 0.0f),
            float4(0.0f, 2.0f * rcpHeight, 0.0f, 0.0f),
            float4(0.0f, 0.0f, -2.0f * rcpDepth, 0.0f),
            float4(-(right + left) * rcpWidth, -(top + bottom) * rcpHeight, -(zFar + zNear) * rcpDepth, 1.0f)
        );
    }

    /**
     * @brief Perspective projection matrix
     * @param fovY Vertical field of view in radians
     * @param aspect Aspect ratio (width/height)
     * @param zNear Near clipping plane
     * @param zFar Far clipping plane
     * @return Perspective projection matrix
     * @note Creates perspective projection with infinite far plane optimization
     */
    inline float4x4 float4x4::perspective(float fovY, float aspect, float zNear, float zFar) noexcept
    {
        const float tanHalfFov = std::tan(fovY * 0.5f);
        const float yScale = 1.0f / tanHalfFov;
        const float xScale = yScale / aspect;
        const float rcpDepth = 1.0f / (zNear - zFar);

        return float4x4(
            float4(xScale, 0.0f, 0.0f, 0.0f),
            float4(0.0f, yScale, 0.0f, 0.0f),
            float4(0.0f, 0.0f, (zFar + zNear) * rcpDepth, -1.0f),
            float4(0.0f, 0.0f, 2.0f * zFar * zNear * rcpDepth, 0.0f)
        );
    }

    /**
     * @brief Look-at view matrix
     * @param eye Camera position
     * @param target Target position
     * @param up Up vector
     * @return View matrix
     * @note Creates camera view matrix looking from eye to target
     * @note Up vector defines camera orientation
     */
    inline float4x4 float4x4::look_at(const float3& eye, const float3& target, const float3& up) noexcept
    {
        float3 z = (target - eye).normalize();
        float3 x = cross(up, z).normalize();
        float3 y = cross(z, x);

        return float4x4(
            float4(x.x, y.x, z.x, 0),
            float4(x.y, y.y, z.y, 0),
            float4(x.z, y.z, z.z, 0),
            float4(-dot(x, eye), -dot(y, eye), -dot(z, eye), 1)
        );
    }

    /**
     * @brief TRS matrix (Translation * Rotation * Scale)
     * @param translation Translation vector
     * @param rotation Rotation quaternion
     * @param scale Scale vector
     * @return TRS transformation matrix
     * @note Composite transformation matrix for 3D objects
     * @note Order: translation * rotation * scale
     */
    inline float4x4 float4x4::TRS(const float3& _translation, const quaternion& rotation, const float3& scale) noexcept
    {
        // Прямое построение вместо умножения матриц
        float4x4 rot_mat = float4x4(rotation);

        return float4x4(
            float4(rot_mat.col0_.xyz() * scale.x, 0),
            float4(rot_mat.col1_.xyz() * scale.y, 0),
            float4(rot_mat.col2_.xyz() * scale.z, 0),
            float4(_translation, 1)
        );
    }

    // ============================================================================
    // Access Operators
    // ============================================================================

    /**
     * @brief Access column by index
     * @param colIndex Column index (0, 1, 2, or 3)
     * @return Reference to column
     * @note Column-major storage: [col][row]
     */
    inline float4& float4x4::operator[](int colIndex) noexcept
    {
        return (colIndex == 0) ? col0_ : (colIndex == 1) ? col1_ : (colIndex == 2) ? col2_ : col3_;
    }

    /**
     * @brief Access column by index (const)
     * @param colIndex Column index (0, 1, 2, or 3)
     * @return Const reference to column
     */
    inline const float4& float4x4::operator[](int colIndex) const noexcept
    {
        return (colIndex == 0) ? col0_ : (colIndex == 1) ? col1_ : (colIndex == 2) ? col2_ : col3_;
    }

    /**
     * @brief Access element by row and column (column-major)
     * @param row Row index (0, 1, 2, or 3)
     * @param col Column index (0, 1, 2, or 3)
     * @return Reference to element
     * @note Column-major: [col][row]
     */
    inline float& float4x4::operator()(int row, int col) noexcept
    {
        return (*this)[col][row]; // column-major: [col][row]
    }

    /**
     * @brief Access element by row and column (const)
     * @param row Row index (0, 1, 2, or 3)
     * @param col Column index (0, 1, 2, or 3)
     * @return Const reference to element
     */
    inline const float& float4x4::operator()(int row, int col) const noexcept
    {
        return (*this)[col][row]; // column-major: [col][row]
    }

    // ============================================================================
    // Column and Row Accessors
    // ============================================================================

    /**
     * @brief Get column 0
     * @return First column
     */
    inline float4 float4x4::col0() const noexcept { return col0_; }

    /**
     * @brief Get column 1
     * @return Second column
     */
    inline float4 float4x4::col1() const noexcept { return col1_; }

    /**
     * @brief Get column 2
     * @return Third column
     */
    inline float4 float4x4::col2() const noexcept { return col2_; }

    /**
     * @brief Get column 3
     * @return Fourth column (translation/perspective)
     */
    inline float4 float4x4::col3() const noexcept { return col3_; }

    inline float4 float4x4::row0() const noexcept
    {
        return float4(col0_.x, col1_.x, col2_.x, col3_.x);
    }

    inline float4 float4x4::row1() const noexcept
    {
        return float4(col0_.y, col1_.y, col2_.y, col3_.y);
    }

    inline float4 float4x4::row2() const noexcept
    {
        return float4(col0_.z, col1_.z, col2_.z, col3_.z);
    }

    inline float4 float4x4::row3() const noexcept
    {
        return float4(col0_.w, col1_.w, col2_.w, col3_.w);
    }

    /**
     * @brief Set column 0
     * @param col New column values
     */
    inline void float4x4::set_col0(const float4& col) noexcept { col0_ = col; }

    /**
     * @brief Set column 1
     * @param col New column values
     */
    inline void float4x4::set_col1(const float4& col) noexcept { col1_ = col; }

    /**
     * @brief Set column 2
     * @param col New column values
     */
    inline void float4x4::set_col2(const float4& col) noexcept { col2_ = col; }

    /**
     * @brief Set column 3
     * @param col New column values
     */
    inline void float4x4::set_col3(const float4& col) noexcept { col3_ = col; }

    /**
     * @brief Set row 0
     * @param row New row values
     */
    inline void float4x4::set_row0(const float4& row) noexcept
    {
        col0_.x = row.x;
        col1_.x = row.y;
        col2_.x = row.z;
        col3_.x = row.w;
    }

    /**
     * @brief Set row 1
     * @param row New row values
     */
    inline void float4x4::set_row1(const float4& row) noexcept
    {
        col0_.y = row.x;
        col1_.y = row.y;
        col2_.y = row.z;
        col3_.y = row.w;
    }

    /**
     * @brief Set row 2
     * @param row New row values
     */
    inline void float4x4::set_row2(const float4& row) noexcept
    {
        col0_.z = row.x;
        col1_.z = row.y;
        col2_.z = row.z;
        col3_.z = row.w;
    }

    /**
     * @brief Set row 3
     * @param row New row values
     */
    inline void float4x4::set_row3(const float4& row) noexcept
    {
        col0_.w = row.x;
        col1_.w = row.y;
        col2_.w = row.z;
        col3_.w = row.w;
    }

    // ============================================================================
    // Compound Assignment Operators (SSE Optimized)
    // ============================================================================

    /**
     * @brief Matrix addition assignment
     * @param rhs Right-hand side matrix
     * @return Reference to this matrix
     * @note SSE optimized
     */
    inline float4x4& float4x4::operator+=(const float4x4& rhs) noexcept
    {
        col0_ += rhs.col0_;
        col1_ += rhs.col1_;
        col2_ += rhs.col2_;
        col3_ += rhs.col3_;
        return *this;
    }

    /**
     * @brief Matrix subtraction assignment
     * @param rhs Right-hand side matrix
     * @return Reference to this matrix
     * @note SSE optimized
     */
    inline float4x4& float4x4::operator-=(const float4x4& rhs) noexcept
    {
        col0_ -= rhs.col0_;
        col1_ -= rhs.col1_;
        col2_ -= rhs.col2_;
        col3_ -= rhs.col3_;
        return *this;
    }

    /**
     * @brief Scalar multiplication assignment
     * @param scalar Scalar multiplier
     * @return Reference to this matrix
     * @note SSE optimized
     */
    inline float4x4& float4x4::operator*=(float scalar) noexcept
    {
        col0_ *= scalar;
        col1_ *= scalar;
        col2_ *= scalar;
        col3_ *= scalar;
        return *this;
    }

    /**
     * @brief Scalar division assignment
     * @param scalar Scalar divisor
     * @return Reference to this matrix
     * @note SSE optimized
     */
    inline float4x4& float4x4::operator/=(float scalar) noexcept
    {
        const float inv_scalar = 1.0f / scalar;
        col0_ *= inv_scalar;
        col1_ *= inv_scalar;
        col2_ *= inv_scalar;
        col3_ *= inv_scalar;
        return *this;
    }

    /**
     * @brief Matrix multiplication assignment
     * @param rhs Right-hand side matrix
     * @return Reference to this matrix
     * @note Uses full SSE optimization
     */
    inline float4x4& float4x4::operator*=(const float4x4& rhs) noexcept
    {
        *this = *this * rhs;
        return *this;
    }

    // ============================================================================
    // Unary Operators
    // ============================================================================

    /**
     * @brief Unary plus operator
     * @return Copy of this matrix
     */
    inline float4x4 float4x4::operator+() const noexcept { return *this; }

    /**
     * @brief Unary minus operator
     * @return Negated matrix
     */
    inline float4x4 float4x4::operator-() const noexcept
    {
        return float4x4(-col0_, -col1_, -col2_, -col3_);
    }

    // ============================================================================
    // Matrix Operations (SSE Optimized)
    // ============================================================================

    /**
     * @brief Compute transposed matrix
     * @return Transposed matrix
     * @note SSE optimized 4x4 transpose using shuffle operations
     */
    inline float4x4 float4x4::transposed() const noexcept
    {
        // SSE optimized 4x4 matrix transpose
        const __m128 tmp0 = _mm_shuffle_ps(col0_.get_simd(), col1_.get_simd(), _MM_SHUFFLE(1, 0, 1, 0));
        const __m128 tmp1 = _mm_shuffle_ps(col0_.get_simd(), col1_.get_simd(), _MM_SHUFFLE(3, 2, 3, 2));
        const __m128 tmp2 = _mm_shuffle_ps(col2_.get_simd(), col3_.get_simd(), _MM_SHUFFLE(1, 0, 1, 0));
        const __m128 tmp3 = _mm_shuffle_ps(col2_.get_simd(), col3_.get_simd(), _MM_SHUFFLE(3, 2, 3, 2));

        return float4x4(
            float4(_mm_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(2, 0, 2, 0))),
            float4(_mm_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(3, 1, 3, 1))),
            float4(_mm_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(2, 0, 2, 0))),
            float4(_mm_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(3, 1, 3, 1)))
        );
    }

    /**
     * @brief Compute matrix determinant
     * @return Determinant value
     * @note Uses Laplace expansion for 4x4 determinant
     */
    inline float float4x4::determinant() const noexcept
    {
        const float a = col0_.x, b = col0_.y, c = col0_.z, d = col0_.w;
        const float e = col1_.x, f = col1_.y, g = col1_.z, h = col1_.w;
        const float i = col2_.x, j = col2_.y, k = col2_.z, l = col2_.w;
        const float m = col3_.x, n = col3_.y, o = col3_.z, p = col3_.w;

        // Вычисляем с проверкой на переполнение
        const float kp_lo = k * p - l * o;
        const float jp_ln = j * p - l * n;
        const float jo_kn = j * o - k * n;
        const float ip_lm = i * p - l * m;
        const float io_km = i * o - k * m;
        const float in_jm = i * n - j * m;

        // Проверяем промежуточные результаты
        if (!std::isfinite(kp_lo) || !std::isfinite(jp_ln) || !std::isfinite(jo_kn) ||
            !std::isfinite(ip_lm) || !std::isfinite(io_km) || !std::isfinite(in_jm)) {
            return 0.0f; // Возвращаем 0 при переполнении
        }

        const float term1 = a * (f * kp_lo - g * jp_ln + h * jo_kn);
        const float term2 = b * (e * kp_lo - g * ip_lm + h * io_km);
        const float term3 = c * (e * jp_ln - f * ip_lm + h * in_jm);
        const float term4 = d * (e * jo_kn - f * io_km + g * in_jm);

        if (!std::isfinite(term1) || !std::isfinite(term2) ||
            !std::isfinite(term3) || !std::isfinite(term4)) {
            return 0.0f;
        }

        return term1 - term2 + term3 - term4;
    }

    inline float4x4 float4x4::inverted_affine() const noexcept
    {
        // Для аффинной матрицы M = [R T; 0 1], обратная = [R^-1 -R^-1*T; 0 1]
        // Но наша матрица может иметь масштабирование, поэтому используем общий подход

        const float3 col0_xyz = col0_.xyz();
        const float3 col1_xyz = col1_.xyz();
        const float3 col2_xyz = col2_.xyz();
        const float3 translation = get_translation();

        // Вычисляем определитель 3x3 верхней левой подматрицы
        const float3 cross_12 = col1_xyz.cross(col2_xyz);
        const float det = col0_xyz.dot(cross_12);

        if (std::abs(det) < Constants::Constants<float>::Epsilon) {
            return identity();
        }

        const float inv_det = 1.0f / det;

        // Вычисляем обратную 3x3 матрицу через алгебраические дополнения
        const float3 inv0 = cross_12 * inv_det;
        const float3 inv1 = col2_xyz.cross(col0_xyz) * inv_det;
        const float3 inv2 = col0_xyz.cross(col1_xyz) * inv_det;

        // Обратная трансляция: -inv_upper * translation
        const float3 inv_translation = float3(
            -dot(inv0, translation),
            -dot(inv1, translation),
            -dot(inv2, translation)
        );

        // Собираем обратную 4x4 матрицу
        return float4x4(
            float4(inv0, 0.0f),
            float4(inv1, 0.0f),
            float4(inv2, 0.0f),
            float4(inv_translation, 1.0f)
        );
    }

    /**
     * @brief Compute inverse matrix
     * @return Inverse matrix
     * @note Returns identity if matrix is singular
     * @warning Matrix must be invertible (determinant != 0)
     */
    inline float4x4 float4x4::inverted() const noexcept
    {
        // Проверяем, является ли матрица аффинной и обратимой
        if (is_affine(Constants::Constants<float>::Epsilon)) {
            // Для аффинных матриц используем оптимизированный алгоритм
            float4x4 result = inverted_affine();

            // Проверяем, что результат корректен
            if (!result.approximately_zero(Constants::Constants<float>::Epsilon)) {
                return result;
            }
        }

        // Для не-аффинных или проблемных случаев используем общий алгоритм
        const float det = determinant();

        if (std::abs(det) < Constants::Constants<float>::Epsilon) {
            return identity();
        }

        return adjugate() * (1.0f / det);
    }

    /**
     * @brief Compute adjugate matrix
     * @return Adjugate matrix
     * @note Transpose of cofactor matrix, used for inverse calculation
     */
    inline float4x4 float4x4::adjugate() const noexcept
    {
        const float a = col0_.x, b = col0_.y, c = col0_.z, d = col0_.w;
        const float e = col1_.x, f = col1_.y, g = col1_.z, h = col1_.w;
        const float i = col2_.x, j = col2_.y, k = col2_.z, l = col2_.w;
        const float m = col3_.x, n = col3_.y, o = col3_.z, p = col3_.w;

        // Вычисляем все необходимые определители 2x2
        const float kp_lo = k * p - l * o;
        const float jp_ln = j * p - l * n;
        const float jo_kn = j * o - k * n;
        const float ip_lm = i * p - l * m;
        const float io_km = i * o - k * m;
        const float in_jm = i * n - j * m;

        const float gp_ho = g * p - h * o;
        const float fp_hn = f * p - h * n;
        const float fo_gn = f * o - g * n;
        const float ep_hm = e * p - h * m;
        const float eo_gm = e * o - g * m;
        const float en_fm = e * n - f * m;

        const float gl_hk = g * l - h * k;
        const float fl_hj = f * l - h * j;
        const float fk_gj = f * k - g * j;
        const float el_hi = e * l - h * i;
        const float ek_gi = e * k - g * i;
        const float ej_fi = e * j - f * i;

        // Матрица алгебраических дополнений (cofactor matrix)
        float4x4 cofactor;

        // Первый столбец присоединенной матрицы (транспонированной cofactor)
        cofactor.col0_.x = +(f * kp_lo - g * jp_ln + h * jo_kn);
        cofactor.col0_.y = -(e * kp_lo - g * ip_lm + h * io_km);
        cofactor.col0_.z = +(e * jp_ln - f * ip_lm + h * in_jm);
        cofactor.col0_.w = -(e * jo_kn - f * io_km + g * in_jm);

        // Второй столбец
        cofactor.col1_.x = -(b * kp_lo - c * jp_ln + d * jo_kn);
        cofactor.col1_.y = +(a * kp_lo - c * ip_lm + d * io_km);
        cofactor.col1_.z = -(a * jp_ln - b * ip_lm + d * in_jm);
        cofactor.col1_.w = +(a * jo_kn - b * io_km + c * in_jm);

        // Третий столбец
        cofactor.col2_.x = +(b * gp_ho - c * fp_hn + d * fo_gn);
        cofactor.col2_.y = -(a * gp_ho - c * ep_hm + d * eo_gm);
        cofactor.col2_.z = +(a * fp_hn - b * ep_hm + d * en_fm);
        cofactor.col2_.w = -(a * fo_gn - b * eo_gm + c * en_fm);

        // Четвертый столбец
        cofactor.col3_.x = -(b * gl_hk - c * fl_hj + d * fk_gj);
        cofactor.col3_.y = +(a * gl_hk - c * el_hi + d * ek_gi);
        cofactor.col3_.z = -(a * fl_hj - b * el_hi + d * ej_fi);
        cofactor.col3_.w = +(a * fk_gj - b * ek_gi + c * ej_fi);

        return cofactor;
    }

    /**
     * @brief Compute normal matrix (transpose(inverse(mat3x3)))
     * @return Normal transformation matrix
     * @note Used for transforming normal vectors in 3D graphics
     * @note Extracts 3x3 rotation/scale part and computes inverse transpose
     */
    inline float3x3 float4x4::normal_matrix() const noexcept
    {
        // Извлекаем верхнюю левую 3x3 подматрицу
        float3x3 mat3x3(
            float3(col0_.x, col0_.y, col0_.z),
            float3(col1_.x, col1_.y, col1_.z),
            float3(col2_.x, col2_.y, col2_.z)
        );

        // Нормальная матрица = inverse(матрица) транспонированная
        // Но для эффективности: normal_matrix = transpose(inverse(mat3x3))
        float3x3 inv = mat3x3.inverted();
        return inv.transposed();
    }

    /**
     * @brief Compute matrix trace (sum of diagonal elements)
     * @return Trace value
     */
    inline float float4x4::trace() const noexcept
    {
        return col0_.x + col1_.y + col2_.z + col3_.w;
    }

    /**
     * @brief Extract diagonal elements
     * @return Diagonal as float4 vector
     */
    inline float4 float4x4::diagonal() const noexcept
    {
        return float4(col0_.x, col1_.y, col2_.z, col3_.w);
    }

    /**
     * @brief Compute Frobenius norm (sqrt of sum of squares of all elements)
     * @return Frobenius norm
     */
    inline float float4x4::frobenius_norm() const noexcept
    {
        return std::sqrt(col0_.length_sq() + col1_.length_sq() + col2_.length_sq() + col3_.length_sq());
    }

    // ============================================================================
    // Vector Transformations (SSE Optimized)
    // ============================================================================

    /**
     * @brief Transform 4D vector (matrix * vector)
     * @param vec 4D vector to transform
     * @return Transformed 4D vector
     * @note SSE optimized matrix-vector multiplication
     * @note Handles homogeneous coordinates and perspective division
     */
    inline float4 float4x4::transform_vector(const float4& vec) const noexcept
    {
        // Matrix-vector multiplication: result = mat * vec
        const __m128 v = vec.get_simd();

        const __m128 r0 = _mm_mul_ps(col0_.get_simd(), _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)));
        const __m128 r1 = _mm_mul_ps(col1_.get_simd(), _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)));
        const __m128 r2 = _mm_mul_ps(col2_.get_simd(), _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2)));
        const __m128 r3 = _mm_mul_ps(col3_.get_simd(), _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3)));

        const __m128 result = _mm_add_ps(_mm_add_ps(r0, r1), _mm_add_ps(r2, r3));
        return float4(result);
    }

    /**
     * @brief Transform 3D point (applies translation and perspective)
     * @param point 3D point to transform
     * @return Transformed 3D point
     * @note Applies full homogeneous transformation with perspective division
     * @note For points (w=1), handles translation and perspective correctly
     */
    inline float3 float4x4::transform_point(const float3& point) const noexcept
    {
        const float4 homogenous_point(point, 1.0f);
        const float4 result = transform_vector(homogenous_point);

        // Для аффинных преобразований w=1, деление не нужно
        if (MathFunctions::approximately(result.w, 1.0f, Constants::Constants<float>::Epsilon)) {
            return result.xyz();
        }
        else {
            return result.xyz() / result.w;
        }
    }

    /**
     * @brief Transform 3D vector (ignores translation)
     * @param vec 3D vector to transform
     * @return Transformed 3D vector
     * @note For vectors (w=0), ignores translation component
     * @note Useful for transforming directions and normals
     */
    inline float3 float4x4::transform_vector(const float3& vec) const noexcept
    {
        const float4 result = transform_vector(float4(vec, 0.0f));
        return result.xyz();
    }

    /**
     * @brief Transform 3D direction (normalizes result)
     * @param dir 3D direction to transform
     * @return Transformed and normalized 3D direction
     * @note Transforms direction vector and normalizes the result
     */
    inline float3 float4x4::transform_direction(const float3& dir) const noexcept
    {
        return transform_vector(dir).normalize();
    }

    // ============================================================================
    // Transformation Component Extraction
    // ============================================================================

    /**
     * @brief Extract translation component
     * @return Translation vector
     * @note Returns the XYZ components of the fourth column
     */
    inline float3 float4x4::get_translation() const noexcept
    {
        return float3(col3_.x, col3_.y, col3_.z);
    }

    /**
     * @brief Extract scale component
     * @return Scale vector
     * @note Computes length of each axis vector (columns 0-2)
     */
    inline float3 float4x4::get_scale() const noexcept
    {
        return float3(
            float3(col0_.x, col0_.y, col0_.z).length(),
            float3(col1_.x, col1_.y, col1_.z).length(),
            float3(col2_.x, col2_.y, col2_.z).length()
        );
    }

    /**
     * @brief Extract rotation component as quaternion
     * @return Rotation quaternion
     * @note Removes scaling and extracts pure rotation
     * @note Uses SSE optimization for scale removal
     */
    inline quaternion float4x4::get_rotation() const noexcept
    {
        float3 col0_norm = col0().xyz().normalize();
        float3 col1_norm = col1().xyz().normalize();
        float3 col2_norm = col2().xyz().normalize();

        float3x3 rot_matrix(col0_norm, col1_norm, col2_norm);
        return quaternion::from_matrix(rot_matrix);
    }

    /**
     * @brief Set translation component
     * @param translation New translation vector
     */
    inline void float4x4::set_translation(const float3& translation) noexcept
    {
        col3_.x = translation.x;
        col3_.y = translation.y;
        col3_.z = translation.z;
    }

    /**
     * @brief Set scale component
     * @param scale New scale vector
     * @note Preserves existing rotation and translation
     */
    inline void float4x4::set_scale(const float3& scale) noexcept
    {
        const float3 current_scale = get_scale();
        if (!current_scale.approximately_zero())
        {
            const float3 inv_current_scale = float3(1.0f) / current_scale;
            col0_.set_xyz(col0_.xyz() * inv_current_scale.x * scale.x);
            col1_.set_xyz(col1_.xyz() * inv_current_scale.y * scale.y);
            col2_.set_xyz(col2_.xyz() * inv_current_scale.z * scale.z);
        }
    }

    // ============================================================================
    // Utility Methods
    // ============================================================================

    /**
     * @brief Check if matrix is identity within tolerance
     * @param epsilon Comparison tolerance
     * @return True if matrix is approximately identity
     */
    inline bool float4x4::is_identity(float epsilon) const noexcept
    {
        return col0_.approximately(float4(1, 0, 0, 0), epsilon) &&
            col1_.approximately(float4(0, 1, 0, 0), epsilon) &&
            col2_.approximately(float4(0, 0, 1, 0), epsilon) &&
            col3_.approximately(float4(0, 0, 0, 1), epsilon);
    }

    /**
     * @brief Check if matrix is affine (last row is [0,0,0,1])
     * @param epsilon Comparison tolerance
     * @return True if matrix is affine
     * @note Affine matrices preserve parallel lines (no perspective)
     */
    inline bool float4x4::is_affine(float epsilon) const noexcept
    {
        // Для column-major матрицы аффинность проверяется по последней СТРОКЕ
        // Последняя строка должна быть [0, 0, 0, 1]
        // В column-major это означает: col0.w, col1.w, col2.w, col3.w

        return MathFunctions::approximately(col0_.w, 0.0f, epsilon) &&
            MathFunctions::approximately(col1_.w, 0.0f, epsilon) &&
            MathFunctions::approximately(col2_.w, 0.0f, epsilon) &&
            MathFunctions::approximately(col3_.w, 1.0f, epsilon);
    }

    /**
     * @brief Check if matrix is orthogonal
     * @param epsilon Comparison tolerance
     * @return True if matrix is orthogonal
     * @note Orthogonal matrices have orthonormal basis vectors
     */
    inline bool float4x4::is_orthogonal(float epsilon) const noexcept
    {
        if (!is_affine(epsilon)) return false;

        const float3 col0_xyz = col0_.xyz();
        const float3 col1_xyz = col1_.xyz();
        const float3 col2_xyz = col2_.xyz();

        // Проверяем ортогональность столбцов
        float dot01 = std::abs(dot(col0_xyz, col1_xyz));
        float dot02 = std::abs(dot(col0_xyz, col2_xyz));
        float dot12 = std::abs(dot(col1_xyz, col2_xyz));

        // Проверяем что столбцы ортогональны
        if (dot01 > epsilon || dot02 > epsilon || dot12 > epsilon) {
            return false;
        }

        // Дополнительно проверяем что столбцы имеют единичную длину
        // (для ортонормированных матриц)
        float len0 = col0_xyz.length_sq();
        float len1 = col1_xyz.length_sq();
        float len2 = col2_xyz.length_sq();

        return MathFunctions::approximately(len0, 1.0f, epsilon) &&
            MathFunctions::approximately(len1, 1.0f, epsilon) &&
            MathFunctions::approximately(len2, 1.0f, epsilon);
    }

    /**
     * @brief Check approximate equality with another matrix
     * @param other Matrix to compare with
     * @param epsilon Comparison tolerance
     * @return True if matrices are approximately equal
     */
    inline bool float4x4::approximately(const float4x4& other, float epsilon) const noexcept
    {
        return col0_.approximately(other.col0_, epsilon) &&
            col1_.approximately(other.col1_, epsilon) &&
            col2_.approximately(other.col2_, epsilon) &&
            col3_.approximately(other.col3_, epsilon);
    }

    /**
     * @brief Convert to string representation
     * @return String representation of matrix
     * @note Format: "[row0]\n[row1]\n[row2]\n[row3]"
     */
    inline std::string float4x4::to_string() const
    {
        char buffer[512];
        std::snprintf(buffer, sizeof(buffer),
            "[%8.4f, %8.4f, %8.4f, %8.4f]\n"
            "[%8.4f, %8.4f, %8.4f, %8.4f]\n"
            "[%8.4f, %8.4f, %8.4f, %8.4f]\n"
            "[%8.4f, %8.4f, %8.4f, %8.4f]",
            col0_.x, col1_.x, col2_.x, col3_.x,  // row0
            col0_.y, col1_.y, col2_.y, col3_.y,  // row1  
            col0_.z, col1_.z, col2_.z, col3_.z,  // row2
            col0_.w, col1_.w, col2_.w, col3_.w); // row3
        return std::string(buffer);
    }

    /**
     * @brief Store matrix to column-major array
     * @param data Destination array (must have at least 16 elements)
     * @note Column-major order: [col0.x, col0.y, col0.z, col0.w, col1.x, ...]
     */
    inline void float4x4::to_column_major(float* data) const noexcept
    {
        data[0] = col0_.x; data[1] = col0_.y; data[2] = col0_.z; data[3] = col0_.w;
        data[4] = col1_.x; data[5] = col1_.y; data[6] = col1_.z; data[7] = col1_.w;
        data[8] = col2_.x; data[9] = col2_.y; data[10] = col2_.z; data[11] = col2_.w;
        data[12] = col3_.x; data[13] = col3_.y; data[14] = col3_.z; data[15] = col3_.w;
    }

    /**
     * @brief Store matrix to row-major array
     * @param data Destination array (must have at least 16 elements)
     * @note Row-major order: [row0.x, row0.y, row0.z, row0.w, row1.x, ...]
     */
    inline void float4x4::to_row_major(float* data) const noexcept
    {
        data[0] = col0_.x; data[1] = col1_.x; data[2] = col2_.x; data[3] = col3_.x;
        data[4] = col0_.y; data[5] = col1_.y; data[6] = col2_.y; data[7] = col3_.y;
        data[8] = col0_.z; data[9] = col1_.z; data[10] = col2_.z; data[11] = col3_.z;
        data[12] = col0_.w; data[13] = col1_.w; data[14] = col2_.w; data[15] = col3_.w;
    }

#if defined(MATH_SUPPORT_D3DX)
    /**
     * @brief Convert to D3DXMATRIX
     * @return D3DXMATRIX equivalent
     * @note Converts from internal column-major to DirectX row-major format
     */
    inline float4x4::operator D3DXMATRIX() const noexcept
    {
        D3DXMATRIX result;
        result._11 = col0_.x; result._12 = col1_.x; result._13 = col2_.x; result._14 = col3_.x;
        result._21 = col0_.y; result._22 = col1_.y; result._23 = col2_.y; result._24 = col3_.y;
        result._31 = col0_.z; result._32 = col1_.z; result._33 = col2_.z; result._34 = col3_.z;
        result._41 = col0_.w; result._42 = col1_.w; result._43 = col2_.w; result._44 = col3_.w;
        return result;
    }
#endif

    // ============================================================================
    // Comparison Operators
    // ============================================================================

    /**
     * @brief Equality comparison
     * @param rhs Right-hand side matrix
     * @return True if matrices are approximately equal
     */
    inline bool float4x4::operator==(const float4x4& rhs) const noexcept
    {
        return approximately(rhs);
    }

    /**
     * @brief Inequality comparison
     * @param rhs Right-hand side matrix
     * @return True if matrices are not approximately equal
     */
    inline bool float4x4::operator!=(const float4x4& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    // ============================================================================
    // Binary Operators
    // ============================================================================

    /**
     * @brief Matrix multiplication
     * @param lhs Left-hand side matrix
     * @param rhs Right-hand side matrix
     * @return Product matrix
     * @note Full SSE optimized 4x4 matrix multiplication
     */
    inline float4x4 operator*(const float4x4& lhs, const float4x4& rhs) noexcept
    {
        float4x4 result;

        const __m128 lhs_col0 = lhs.col0_.get_simd();
        const __m128 lhs_col1 = lhs.col1_.get_simd();
        const __m128 lhs_col2 = lhs.col2_.get_simd();
        const __m128 lhs_col3 = lhs.col3_.get_simd();

        // Обрабатываем все 4 столбца rhs одновременно для лучшей производительности
        const __m128 rhs_col0 = rhs.col0_.get_simd();
        const __m128 rhs_col1 = rhs.col1_.get_simd();
        const __m128 rhs_col2 = rhs.col2_.get_simd();
        const __m128 rhs_col3 = rhs.col3_.get_simd();

        // Столбец 0 результата
        __m128 rhs0_x = _mm_shuffle_ps(rhs_col0, rhs_col0, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 rhs0_y = _mm_shuffle_ps(rhs_col0, rhs_col0, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 rhs0_z = _mm_shuffle_ps(rhs_col0, rhs_col0, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 rhs0_w = _mm_shuffle_ps(rhs_col0, rhs_col0, _MM_SHUFFLE(3, 3, 3, 3));

        result.col0_ = float4(_mm_add_ps(
            _mm_add_ps(_mm_mul_ps(lhs_col0, rhs0_x), _mm_mul_ps(lhs_col1, rhs0_y)),
            _mm_add_ps(_mm_mul_ps(lhs_col2, rhs0_z), _mm_mul_ps(lhs_col3, rhs0_w))
        ));

        // Столбец 1 результата
        __m128 rhs1_x = _mm_shuffle_ps(rhs_col1, rhs_col1, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 rhs1_y = _mm_shuffle_ps(rhs_col1, rhs_col1, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 rhs1_z = _mm_shuffle_ps(rhs_col1, rhs_col1, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 rhs1_w = _mm_shuffle_ps(rhs_col1, rhs_col1, _MM_SHUFFLE(3, 3, 3, 3));

        result.col1_ = float4(_mm_add_ps(
            _mm_add_ps(_mm_mul_ps(lhs_col0, rhs1_x), _mm_mul_ps(lhs_col1, rhs1_y)),
            _mm_add_ps(_mm_mul_ps(lhs_col2, rhs1_z), _mm_mul_ps(lhs_col3, rhs1_w))
        ));

        // Столбец 2 результата
        __m128 rhs2_x = _mm_shuffle_ps(rhs_col2, rhs_col2, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 rhs2_y = _mm_shuffle_ps(rhs_col2, rhs_col2, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 rhs2_z = _mm_shuffle_ps(rhs_col2, rhs_col2, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 rhs2_w = _mm_shuffle_ps(rhs_col2, rhs_col2, _MM_SHUFFLE(3, 3, 3, 3));

        result.col2_ = float4(_mm_add_ps(
            _mm_add_ps(_mm_mul_ps(lhs_col0, rhs2_x), _mm_mul_ps(lhs_col1, rhs2_y)),
            _mm_add_ps(_mm_mul_ps(lhs_col2, rhs2_z), _mm_mul_ps(lhs_col3, rhs2_w))
        ));

        // Столбец 3 результата
        __m128 rhs3_x = _mm_shuffle_ps(rhs_col3, rhs_col3, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 rhs3_y = _mm_shuffle_ps(rhs_col3, rhs_col3, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 rhs3_z = _mm_shuffle_ps(rhs_col3, rhs_col3, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 rhs3_w = _mm_shuffle_ps(rhs_col3, rhs_col3, _MM_SHUFFLE(3, 3, 3, 3));

        result.col3_ = float4(_mm_add_ps(
            _mm_add_ps(_mm_mul_ps(lhs_col0, rhs3_x), _mm_mul_ps(lhs_col1, rhs3_y)),
            _mm_add_ps(_mm_mul_ps(lhs_col2, rhs3_z), _mm_mul_ps(lhs_col3, rhs3_w))
        ));

        return result;
    }

    // ============================================================================
    // Useful Constants
    // ============================================================================

    /**
     * @brief Identity matrix constant
     */
    inline const float4x4 float4x4_Identity = float4x4::identity();

    /**
     * @brief Zero matrix constant
     */
    inline const float4x4 float4x4_Zero = float4x4::zero();
} // namespace Math

#endif // MATH_FLOAT4X4_INL
