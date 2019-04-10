// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "TransformationEstimation.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/Utility/Eigen.h>
#include <iostream>

namespace Eigen { 
 
#ifndef EIGEN_PARSED_BY_DOXYGEN

    // These helpers are required since it allows to use mixed types as parameters
    // for the Umeyama. The problem with mixed parameters is that the return type
    // cannot trivially be deduced when float and double types are mixed.
    namespace internal {

        // Compile time return type deduction for different MatrixBase types.
        // Different means here different alignment and parameters but the same underlying
        // real scalar type.
        template<typename MatrixType, typename OtherMatrixType>
            struct umeyama_transform_matrix_type2
            {
                enum {
                    MinRowsAtCompileTime = EIGEN_SIZE_MIN_PREFER_DYNAMIC(MatrixType::RowsAtCompileTime, OtherMatrixType::RowsAtCompileTime),

                    // When possible we want to choose some small fixed size value since the result
                    // is likely to fit on the stack. So here, EIGEN_SIZE_MIN_PREFER_DYNAMIC is not what we want.
                    HomogeneousDimension = int(MinRowsAtCompileTime) == Dynamic ? Dynamic : int(MinRowsAtCompileTime)+1
                };

                typedef Matrix<typename traits<MatrixType>::Scalar,
                        HomogeneousDimension,
                        HomogeneousDimension,
                        AutoAlign | (traits<MatrixType>::Flags & RowMajorBit ? RowMajor : ColMajor),
                        HomogeneousDimension,
                        HomogeneousDimension
                            > type;
            };

    }

#endif

    // Similar to the original Eigen::Umeyama  https://eigen.tuxfamily.org/dox/Umeyama_8h_source.html
    template <typename Derived, typename OtherDerived>
    Eigen::Matrix4d umeyama_constrained(const Eigen::MatrixXd& _src, const Eigen::MatrixXd& _dst, bool with_scaling = true) {
        typedef typename Eigen::internal::umeyama_transform_matrix_type2<Derived, OtherDerived>::type TransformationMatrixType;
        typedef typename Eigen::internal::traits<TransformationMatrixType>::Scalar Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;

        enum { Dimension = EIGEN_SIZE_MIN_PREFER_DYNAMIC(Derived::RowsAtCompileTime, OtherDerived::RowsAtCompileTime) };

        typedef Eigen::Matrix<Scalar, Dimension, 1> VectorType;
        typedef Eigen::Matrix<Scalar, Dimension, Dimension> MatrixType;
        typedef typename Eigen::internal::plain_matrix_type_row_major<Derived>::type RowMajorMatrixType;

        const Index m = _src.rows(); // dimension
        const Index n = _src.cols(); // number of measurements

        Eigen::MatrixXd src(m, n);
        src << _src.topRows(2), Eigen::MatrixXd::Zero(1, n);
        Eigen::MatrixXd dst(m, n);
        dst << _dst.topRows(2), Eigen::MatrixXd::Zero(1, n);

        // required for demeaning ...
        const RealScalar one_over_n = RealScalar(1) / static_cast<RealScalar>(n);

        // computation of mean
        const VectorType src_mean = src.rowwise().sum() * one_over_n;
        const VectorType dst_mean = dst.rowwise().sum() * one_over_n;

        // demeaning of src and dst points
        const RowMajorMatrixType src_demean = src.colwise() - src_mean;
        const RowMajorMatrixType dst_demean = dst.colwise() - dst_mean;

        // Eq. (36)-(37)
        const Scalar src_var = src_demean.rowwise().squaredNorm().sum() * one_over_n;

        // Eq. (38)
        const MatrixType sigma = one_over_n * dst_demean * src_demean.transpose();

        Eigen::JacobiSVD<MatrixType> svd(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);

        // Initialize the resulting transformation with an identity matrix...
        TransformationMatrixType Rt = TransformationMatrixType::Identity(m+1,m+1);

        // Eq. (39)
        VectorType S = VectorType::Ones(m);

        if  ( svd.matrixU().determinant() * svd.matrixV().determinant() < 0 ) {
            // S(m-1) = -1;
            Rt.col(m).head(m) = dst_mean-src_mean;
            return Rt;
        }

        // Eq. (40) and (43)
        Rt.block(0,0,m,m).noalias() = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();

        if (with_scaling)
        {
            // Eq. (42)
            const Scalar c = Scalar(1)/src_var * svd.singularValues().dot(S);

            // Eq. (41)
            Rt.col(m).head(m) = dst_mean;
            Rt.col(m).head(m).noalias() -= c*Rt.topLeftCorner(m,m)*src_mean;
            Rt.block(0,0,m,m) *= c;
        }
        else
        {
            Rt.col(m).head(m) = dst_mean;
            Rt.col(m).head(m).noalias() -= Rt.topLeftCorner(m,m)*src_mean;
        }
        return Rt;
    }
}

namespace open3d {
namespace registration {

double TransformationEstimationPointToPoint::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty()) return 0.0;
    double err = 0.0;
    for (const auto &c : corres) {
        err += (source.points_[c[0]] - target.points_[c[1]]).squaredNorm();
    }
    return std::sqrt(err / (double)corres.size());
}

Eigen::Matrix4d TransformationEstimationPointToPoint::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty()) return Eigen::Matrix4d::Identity();
    Eigen::MatrixXd source_mat(3, corres.size());
    Eigen::MatrixXd target_mat(3, corres.size());
    for (size_t i = 0; i < corres.size(); i++) {
        source_mat.block<3, 1>(0, i) = source.points_[corres[i][0]];
        target_mat.block<3, 1>(0, i) = target.points_[corres[i][1]];
    }
    if (with_constraint_) {
        return Eigen::umeyama_constrained<Eigen::MatrixXd, Eigen::MatrixXd>(source_mat, target_mat, with_scaling_);
    }
    else {
        return Eigen::umeyama(source_mat, target_mat, with_scaling_);
    }
}

double TransformationEstimationPointToPlane::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || target.HasNormals() == false) return 0.0;
    double err = 0.0, r;
    for (const auto &c : corres) {
        r = (source.points_[c[0]] - target.points_[c[1]])
                    .dot(target.normals_[c[1]]);
        err += r * r;
    }
    return std::sqrt(err / (double)corres.size());
}

Eigen::Matrix4d TransformationEstimationPointToPlane::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || target.HasNormals() == false)
        return Eigen::Matrix4d::Identity();

    auto compute_jacobian_and_residual = [&](int i, Eigen::Vector6d &J_r,
                                             double &r) {
        const Eigen::Vector3d &vs = source.points_[corres[i][0]];
        const Eigen::Vector3d &vt = target.points_[corres[i][1]];
        const Eigen::Vector3d &nt = target.normals_[corres[i][1]];
        r = (vs - vt).dot(nt);
        J_r.block<3, 1>(0, 0) = vs.cross(nt);
        J_r.block<3, 1>(3, 0) = nt;
    };

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
    double r2;
    std::tie(JTJ, JTr, r2) =
            utility::ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
                    compute_jacobian_and_residual, (int)corres.size());

    bool is_success;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
            utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);

    return is_success ? extrinsic : Eigen::Matrix4d::Identity();
}

}  // namespace registration
}  // namespace open3d
