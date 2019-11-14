class IMNN():
    def __init__(self, n_s, numerical=False, dθ=None):
        self._FLOATX = tf.float32

    def get_covariance(self, x):
        μ = tf.reduce_mean(
            x,
            axis=0,
            keepdims=True,
            name="mean")
        μ_offset = tf.subtract(
            x,
            μ,
            name="centred_mean")
        C = tf.divide(
            tf.einsum(
                "ij,ik->jk",
                μ_offset,
                μ_offset,
                name="unnormalised_covariance"),
            self.n_sm1,
            name="covariance")
        return C, μ_offset

    def get_numerical_derivative_mean(self, x_m, x_p):
        return tf.reduce_mean(
            tf.einsum(
                "ijk,j->ijk",
                tf.subtract(
                    x_p,
                    x_m,
                    name="numerical_summary_difference"),
                self.dθ,
                name="numerical_derivative_summaries_wrt_parameters"),
            axis=0,
            name="numerical_derivative_mean_wrt_parameters")

    def get_Fisher(self, Cinv, dμdθ):
        F = tf.linalg.band_part(
            tf.einsum(
                "ij,kj->ik",
                dμdθ,
                self.score,
                name="half_fisher"),
            0,
            -1,
            name="triangle_fisher")
        return tf.multiply(
            0.5,
            tf.add(
                F,
                tf.transpose(
                    F,
                    perm=[1, 0],
                    name="transposed_fisher"),
                name="double_fisher"),
            name="fisher")

    def get_covariance_derivative(self, μ_offset):
        return tf.divide(
            tf.reduce_sum(
                tf.add(
                    tf.einsum(
                        "ijkl,im->ijkml",
                        self.dΔμdx,
                        μ_offset,
                        name="centred_mean_derivative_first_half"),
                    tf.einsum(
                        "ij,iklm->ikjlm",
                        μ_offset,
                        self.dΔμdx,
                        name="centred_mean_derivative_second_half"),
                    name="centred_mean_derivative"),
                axis=0,
                name="unnormalised_covariance_derivative"),
            tf.subtract(ns, 1.),
            name="covariance_derivative")

    def get_fisher_derivative(self, Cinv, dμdθ, dCdx):
        dFdx = tf.linalg.band_part(
            tf.einsum(
                "ij,kljm->kmil",
                dμdθ,
                tf.einsum(
                    "ij,kljm->klim",
                    Cinv,
                    tf.einsum(
                        "ijkl,mk->imjl",
                        dCdx,
                        self.score,
                        name="covariance_derivative_score"),
                    name="inverse_covariance_covariance_derivative_score"),
                name="half_derivative_of_fisher_wrt_summaries"),
            0,
            -1,
            name="triangle_derivative_of_fisher_wrt_summaries")
        dFdx = tf.multiply(
            -0.5,
            tf.add(
                dFdx,
                tf.transpose(
                    dFdx,
                    perm=[0, 1, 3, 2],
                    name="transposed_derivative_of_fisher_wrt_summaries"),
                name="double_derivative_of_fisher_wrt_summaries"),
            name="derivative_of_fisher_wrt_summaries")

        dFdx_m_a = tf.linalg.band_part(
            tf.einsum(
                "ijklm,nk->imljn",
                -ddμdθdx,
                self.score), 0, -1)
        dFdx_m_b = tf.linalg.band_part(
            tf.einsum(
                "ij,kljmn->knmil",
                dμdθ,
                tf.einsum(
                    "ij,kljmn->klimn",
                    Cinv,
                    -ddμdθdx)), 0, -1)
        dFdx_m = tf.multiply(
            0.5,
            tf.add(
                tf.add(
                    dFdx_m_a,
                    tf.transpose(
                        dFdx_m_a,
                        perm=[0, 1, 2, 4, 3])),
                tf.add(
                    dFdx_m_b,
                    tf.transpose(
                        dFdx_m_b,
                        perm=[0, 1, 2, 4, 3]))),
            name="derivative_of_fisher_wrt_lower_summaries")

        dFdx_p_a = tf.linalg.band_part(
            tf.einsum(
                "ijklm,nk->imljn",
                ddμdθdx,
                Cinvdμdθ), 0, -1)
        dFdx_p_b = tf.linalg.band_part(
            tf.einsum(
                "ij,kljmn->knmil",
                dμdθ,
                tf.einsum(
                    "ij,kljmn->klimn",
                    Cinv,
                    ddμdθdx)), 0, -1)
        dFdx_p = tf.multiply(
            0.5,
            tf.add(
                tf.add(
                    dFdx_p_a,
                    tf.transpose(
                        dFdx_p_a,
                        perm=[0, 1, 2, 4, 3])),
                tf.add(
                    dFdx_p_b,
                    tf.transpose(
                        dFdx_p_b,
                        perm=[0, 1, 2, 4, 3]))),
            name="derivative_of_fisher_wrt_upper_summaries")
    return dFdx, dFdx_m, dFdx_p

    def get_log_det_fisher_derivative(self, F, Cinv, dμdθ, dCdx):
        Finv = tf.linalg.inv(F)
        dFdx, dFdx_m, dFdx_p = self.get_fisher_derivative(Cinv, dμdθ, dCdx)

        dΛdx = -tf.linalg.trace(
            tf.einsum(
                "ij,kljm->klim",
                Finv,
                dFdx),
            name="derivative_of_logdetfisher_wrt_summaries")
        dΛdx_m = -tf.linalg.trace(
            tf.einsum(
                "ij,klmjn->klmin",
                Finv,
                dFdx_m),
            name="derivative_of_logdetfisher_wrt_lower_summaries")
        dΛdx_p = -tf.linalg.trace(
            tf.einsum(
                "ij,klmjn->klmin",
                Finv,
                dFdx_p),
            name="derivative_of_logdetfisher_wrt_upper_summaries")
    return dΛdx, dΛdx_m, dΛdx_p

    def get_regularisation(self, C, Cinv):
        I = tf.eye(n_summaries)
        CmI = tf.subtract(C, I)
        CinvmI = tf.subtract(Cinv, I)
        Λ_2 = tf.multiply(0.5,
            tf.add(
                tf.square(
                    tf.norm(CmI,
                            ord="fro",
                            axis=(0, 1))),
                tf.square(
                    tf.norm(CinvmI,
                            ord="fro",
                            axis=(0, 1)))),
            name="regulariser")
        return Λ_2, CmI

    def get_regularisation_strength(self, Λ_2, λ, α):
        rate = tf.multiply(-α, Λ_2)
        e_rate = tf.exp(rate)
        r = tf.divide(
                tf.multiply(
                    λ,
                    Λ_2),
                tf.add(
                    Λ_2,
                    e_rate))
        drdΛ_2 = tf.multiply(
            r,
            tf.add(
                1.,
                tf.divide(
                    tf.multiply(
                        tf.add(
                            1.,
                            tf.multiply(
                                α,
                                Λ_2)),
                        e_rate),
                    tf.add(
                        Λ_2,
                        e_rate))))
        return r, drdΛ_2

    def get_regularisation_derivative(self, C, Cinv, dCdx, λ, α):
        Λ_2, CmI = self.get_regularisation(C, Cinv)
        r, drdΛ_2= self.get_regularisation_strength(Λ_2, λ, α)
        Cinv2 = tf.einsum(
            "ij,jk->ik",
            Cinv,
            Cinv)
        Cinv3 = tf.einsum(
            "ij,jk->ik",
            Cinv2,
            Cinv)
        dΛ_2dx = tf.multiply(
            tf.add(
                tf.multiply(
                    Λ_2,
                    drdΛ_2),
                r),
            tf.linalg.trace(
                tf.einsum(
                    "ij,kjlm->kilm",
                    tf.add(
                        CmI,
                        tf.subtract(
                            Cinv2,
                            Cinv3)),
                    dCdx)))
        return dΛ_2dx, r

    self.n_sm1 = tf.subtract(
        tf.cast(
            n_s,
            self._FLOATX,
            name="number_of_simulations_float"),
        name="number_of_simulations_minus_1")
    if numerical:
        self.dθ = tf.Variable(
            1. / (2. * dθ),
            dtype=self._FLOATX,
            trainable=False,
            name="varied_parameters")

    dxdx = tf.einsum(
        "ij,kl->ijkl",
        tf.eye(n_s, n_s),
        tf.eye(n_summaries, n_summaries),
        name="derivative_of_summaries_wrt_summaries")
    dμdx = tf.reduce_mean(
        dxdx,
        axis=0,
        keepdims=True,
        name="derivative_of_mean_summaries_wrt_summaries")
    self.dΔμdx = tf.subtract(
        dxdx,
        dμdx,
        name="derivative_of_centred_mean_summaries_wrt_summaries")

    #Numerical only?
    dxadxb = tf.einsum(
        "ij,kl,mn->ijklmn",
        tf.eye(n_d, n_d),
        tf.eye(n_params, n_params),
        tf.eye(n_summaries, n_summaries),
        name="derivative_of_summaries_wrt_summaries_for_derivatives")
    self.ddμdθdx = tf.reduce_mean(
        tf.einsum(
            "ijklmn,l->ijkmnl",
            dxadxb,
            self.dθ,
            name="derivative_of_summaries_wrt_summaries_and_parameters"),
        axis=0,
        name="derivative_of_mean_summaries_wrt_summaries_and_parameters")



    self.x = tf.zeros()
    self.x_m = tf.zeros()
    self.x_p = tf.zeros()
    self.dxdd = tf.zeros()

    self.score = tf.einsum(
        "ij,kj->ki",
        Cinv,
        dμdθ,
        name="score")
