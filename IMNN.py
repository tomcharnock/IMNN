import tensorflow as tf
import utils


class IMNN():
    def __init__(self, n_params, n_summaries, num_covariance_simulations,
                 num_derivative_simulations, dtype=tf.float32):
        if dtype == tf.float64:
            self.dtype = tf.float32
            self.itype = tf.int64
        else:
            self.dtype = tf.float32
            self.itype = tf.int32
        self.n_params = utils.positive_integer(n_params, "n_params")
        self.n_summaries = utils.positive_integer(n_summaries, "n_summaries")
        self.n_s = utils.positive_integer(
            num_covariance_simulations,
            "num_covariance_simulations")
        self.n_d = utils.positive_integer(
            num_derivative_simulations,
            "num_derivative_simulations")

        self.single_dataset = utils.check_num_datasets(self.n_s, self.n_d)

        self.dΔμ_dx = self.get_dΔμ_dx()
        self.n_sm1 = self.get_n_s_minus_1()
        self.identity = tf.eye(self.n_summaries)

        self.loop_sims = None
        self.n_batch = None
        self.n_d_batch = None
        self.numerical = None
        self.use_external = None
        self.sims_at_once = None

        self.initialise_history()

        self.model = None
        self.optimiser = None

        self.input_shape = None
        self.θ_fid = None
        self.δθ = None
        self.data_iterator = None
        self.data_derivative_iterator = None
        self.indices_iterator = None
        self.derivative_indices_iterator = None
        self.test_θ_fid = None
        self.test_δθ = None
        self.test_data_iterator = None
        self.test_data_derivative_iterator = None

        self.d2μ_dθdx = None
        # self.test_d2μ_dθdx = None

    def initialise_history(self):
        self.history = {
            "det_F": [],
            "val_det_F": [],
            "det_C": [],
            "val_det_C": [],
            "det_Cinv": [],
            "val_det_Cinv": [],
            "dμdθ": [],
            "val_dμdθ": [],
            "reg": [],
            "val_reg": [],
            "r": [],
            "val_r": []
        }

    def load_fiducial(self, θ_fid, train):
        utils.fiducial_check(θ_fid, self.n_params)
        if train:
            self.θ_fid = tf.Variable(θ_fid, dtype=self.dtype, trainable=False,
                                     name="fiducial")
        else:
            self.test_θ_fid = tf.Variable(θ_fid, dtype=self.dtype,
                                          trainable=False,
                                          name="test_fiducial")

    def check_derivative(self, dd_dθ, δθ, train):
        # TO DO - CHECK SHAPE OF DERIVATIVE
        numerical = utils.bool_none(δθ)
        if numerical:
            utils.delta_check(δθ, self.n_params)
        if train:
            if numerical:
                self.δθ = tf.Variable(
                    1. / (2. * δθ), dtype=self.dtype, trainable=False,
                    name="delta_theta")
                self.d2μ_dθdx = self.get_d2μ_dθdx(self.δθ)
            self.numerical = numerical
        else:
            if numerical:
                self.test_δθ = tf.Variable(
                    1. / (2. * δθ), dtype=self.dtype, trainable=False,
                    name="delta_theta")
                # self.test_d2μ_dθdx = self.get_d2μ_dθdx(self.test_δθ)
            self.test_numerical = numerical
        return numerical

    def to_loop_sims(self, sims_at_once, train):
        loop_sims = utils.bool_none(sims_at_once)
        if train:
            self.loop_sims = loop_sims
        else:
            self.test_loop_sims = loop_sims
        return loop_sims

    def use_external_summaries(self, external_summaries, train):
        use_summaries = utils.bool_none(external_summaries)
        if train:
            self.use_summaries = use_summaries
        else:
            self.test_use_summaries = use_summaries
        return use_summaries

    def calculate_batch_size(self, shape, size, train, derivative=False):
        n_batch = shape / size
        if float(int(n_batch)) == n_batch:
            n_batch = int(n_batch)
        else:
            utils.batch_warning()
        if train:
            if not derivative:
                self.n_batch = n_batch
            else:
                self.n_d_batch = n_batch
        else:
            if not derivative:
                self.test_n_batch = n_batch
            else:
                self.test_n_d_batch = n_batch
        return n_batch

    def get_data_shape(self, data):
        self.input_shape = data.shape[1:]

    def build_dataset(self, data, batchsize=None, shufflesize=None):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if batchsize is not None:
            dataset = dataset.batch(batchsize)
        elif shufflesize is not None:
            dataset = dataset.shuffle(shufflesize)
        return dataset.repeat()

    def setup_dataset(self, θ_fid, d, dd_dθ, δθ=None, external_summaries=None,
                      external_derivatives=None, sims_at_once=None,
                      train=True):

        self.load_fiducial(θ_fid, train)
        self.get_data_shape(d)
        numerical = self.check_derivative(dd_dθ, δθ, train)
        loop_sims = self.to_loop_sims(sims_at_once, train)
        use_summaries = self.use_external_summaries(external_summaries, train)
        n_batch = self.calculate_batch_size(d.shape[0], self.n_s, train)

        if (not self.single_dataset) and (not use_summaries):
            data = d
        else:
            data = (d,)

        if self.single_dataset:
            utils.size_check(dd_dθ.shape[0], d.shape[0],
                             "dd_dθ", "d")
            n_d_batch = n_batch
            data += (dd_dθ,)
        else:
            if numerical:
                n_d_batch = self.calculate_batch_size(dd_dθ.shape[0], self.n_d,
                                                      train, derivative=True)
            else:
                utils.numerical_size_check(dd_dθ.shape[0], d.shape[0],
                                           numerical)
                n_d_batch = n_batch
            d_data = (dd_dθ,)

        if use_summaries:
            utils.size_check(external_summaries.shape[0], d.shape[0],
                             "external_summaries", "d")
            data += (external_summaries,)
            if self.single_dataset:
                utils.size_check(external_derivatives.shape[0], d.shape[0],
                                 "external_derivatives", "d")
                data += (external_derivatives,)
            else:
                if numerical:
                    utils.size_check(external_derivatives.shape[0],
                                     dd_dθ.shape[0], "external_derivatives",
                                     "dd_dθ")
                    d_data += (external_derivatives,)

        dataset = self.build_dataset(data, batchsize=self.n_s,
                                     shufflesize=self.n_s * n_batch)

        if not self.single_dataset:
            d_dataset = self.build_dataset(d_data, batchsize=self.n_d,
                                           shufflesize=self.n_d * n_d_batch)
        if loop_sims:
            def loop_batch(*x):
                return self.loop_batch(sims_at_once, *x)

            ind = tf.expand_dims(tf.range(self.n_s, dtype=self.itype), 1)
            indices = self.build_dataset(ind, batchsize=sims_at_once)
            if self.single_dataset:
                dataset = dataset.map(loop_batch)
            else:
                d_ind = tf.expand_dims(tf.range(self.n_d, dtype=self.itype), 1)
                d_indices = self.build_dataset(d_ind, batchsize=sims_at_once)
                if use_summaries:
                    dataset = dataset.map(loop_batch)
                    d_dataset = dataset.map(loop_batch)
                else:
                    dataset = dataset.map(
                        lambda x: (tf.data.Dataset.from_tensor_slices(x)
                                   .batch(sims_at_once).repeat(2)))
                    d_dataset = d_dataset.map(
                        lambda x: (tf.data.Dataset.from_tensor_slices(x)
                                   .batch(sims_at_once).repeat(2)))

        if train:
            self.data_iterator = iter(dataset)
            if loop_sims:
                self.indices_iterator = iter(indices)
            if not self.single_dataset:
                self.data_derivative_iterator = iter(d_dataset)
                if sims_at_once is not None:
                    self.derivative_indices_iterator = iter(d_indices)
        else:
            self.test_data_iterator = iter(dataset)
            if loop_sims:
                self.test_indices_iterator = iter(indices)
            if not self.single_dataset:
                self.test_data_derivative_iterator = iter(d_dataset)
                if sims_at_once is not None:
                    self.test_derivative_indices_iterator = iter(d_indices)

    def loop_batch(self, sims_at_once, *x):
        new_batch = tuple()
        for i in range(len(x)):
            new_batch += (tf.data.Dataset.from_tensor_slices(x[i])
                          .batch(self.sims_at_once)
                          .repeat(2),)
        return new_batch

    def set_model(self, model, optimiser):
        utils.check_model(self.n_params, self.n_summaries)
        self.model = model
        self.optimiser = optimiser

    def get_n_s_minus_1(self):
        return tf.subtract(
            tf.cast(
                self.n_s,
                self.dtype,
                name="number_of_simulations_float"),
            1.,
            name="number_of_simulations_minus_1")

    def get_dΔμ_dx(self):
        dx_dx = tf.einsum(
            "ij,kl->ijkl",
            tf.eye(self.n_s, self.n_s),
            tf.eye(self.n_summaries, self.n_summaries),
            name="derivative_of_summaries_wrt_summaries")
        dμ_dx = tf.reduce_mean(dx_dx, axis=0, keepdims=True,
                               name="derivative_of_mean_x_wrt_x")
        return tf.subtract(dx_dx, dμ_dx,
                           name="derivative_of_diff_mean_x_wrt_x")

    def get_d2μ_dθdx(self, δθ):
        dxa_dxb = tf.einsum(
            "ij,kl,mn->ijklmn",
            tf.eye(self.n_d, self.n_d),
            tf.eye(self.n_params, self.n_params),
            tf.eye(self.n_summaries, self.n_summaries),
            name="derivative_of_x_wrt_x_for_derivatives")
        return tf.reduce_mean(
            tf.einsum(
                "ijklmn,l->ijkmnl",
                dxa_dxb,
                δθ,
                name="derivative_of_x_wrt_x_and_parameters"),
            axis=0,
            name="derivative_of_mean_x_wrt_x_and_parameters")

    def fit(self, n_iterations, reset=False, validate=False):
        import tqdm  # Do notebook checking
        if reset:
            self.initialise_history()
            self.model.reset_states()
        bar = tqdm.tnrange(n_iterations, desc="Iterations")
        for iterations in bar:
            for batch in range(self.n_batch):
                print(batch)
                if self.single_dataset:
                    temp = self.simple_train_iteration(self.data_iterator)
                else:
                    temp = self.simple_train_iteration(
                        self.data_iterator,
                        self.data_derivative_iterator)
            self.history["det_F"].append(temp[0].numpy())
            self.history["det_C"].append(temp[1].numpy())
            self.history["det_Cinv"].append(temp[2].numpy())
            self.history["dμdθ"].append(temp[3].numpy())
            self.history["reg"].append(temp[4].numpy())
            self.history["r"].append(temp[5].numpy())
            if validate:
                for batch in range(self.test_n_batch):
                    if self.single_dataset:
                        if self.numerical:
                            temp = self.numerical_validate(
                                self.test_data_iterator)
                        else:
                            print("Not yet implemented")
                            break
                    else:
                        if self.numerical:
                            temp = self.numerical_validate(
                                self.test_data_iterator,
                                self.test_data_derivative_iterator)
                        else:
                            print("Not yet implemented")
                            break
                self.history["val_det_F"].append(temp[0].numpy())
                self.history["val_det_C"].append(temp[1].numpy())
                self.history["val_det_Cinv"].append(temp[2].numpy())
                self.history["val_dμdθ"].append(temp[3].numpy())
                self.history["val_reg"].append(temp[4].numpy())
                self.history["val_r"].append(temp[5].numpy())
                bar.set_postfix(
                    det_F=self.history["det_F"][-1],
                    det_C=self.history["det_C"][-1],
                    det_Cinv=self.history["det_Cinv"][-1],
                    r=self.history["r"][-1],
                    val_det_F=self.history["val_det_F"][-1],
                    val_det_C=self.history["val_det_C"][-1],
                    val_det_Cinv=self.history["val_det_Cinv"][-1],
                    val_r=self.history["val_r"][-1])
            else:
                bar.set_postfix(
                    det_F=self.history["det_F"][-1],
                    det_C=self.history["det_C"][-1],
                    det_Cinv=self.history["det_Cinv"][-1],
                    r=self.history["r"][-1])

    @tf.function
    def simple_train_iteration(self, data_iterator, derivative_iterator=None):
        with tf.GradientTape(persistent=True) as tape:
            print("Get data")
            if self.single_dataset:
                if self.use_summaries:
                    d, dd_dθ, s, ds_dθ = next(data_iterator)
                else:
                    d, dd_dθ = next(data_iterator)
            else:
                if self.use_summaries:
                    d, s = next(data_iterator)
                    dd_dθ, ds_dθ = next(derivative_iterator)
                else:
                    d = next(data_iterator)
                    dd_dθ = next(derivative_iterator)
            print("Pass data through model")
            if not self.numerical:
                tape.watch(d)
            else:
                dx_dθ = self.model(dd_dθ)
            x = self.model(d)
        print("Calculate jacobian")
        dx_dw = tape.jacobian(x, self.model.variables)
        if not self.numerical:
            dx_dd = tape.batch_jacobian(x, d)
        else:
            ddx_dwdθ = tape.jacobian(dx_dθ, self.model.variables)
        del tape
        print("Get covariance")
        C, Δμ = self.get_covariance(x)
        dC_dx = self.get_covariance_derivative(Δμ)
        Cinv = tf.linalg.inv(C)
        print("Get derivative of mean")
        if not self.numerical:
            dμ_dθ = self.get_chain_derivative_mean(dx_dd, dd_dθ)
        else:
            dμ_dθ = self.get_numerical_derivative_mean(dx_dθ, self.δθ)
        score = self.get_score(Cinv, dμ_dθ)
        print("Calculate Fisher")
        F = self.get_fisher(Cinv, dμ_dθ, score)
        Finv = tf.linalg.inv(F)
        print("Calculate loss")
        dΛ_dx = self.get_loss(Finv, Cinv, dμ_dθ, score, dC_dx)
        if not self.numerical:
            print("To do")
        else:
            ddΛ_dxdθ = tf.stack(
                [self.get_num_loss(Finv, Cinv, dμ_dθ, score, dC_dx, -1.),
                 self.get_num_loss(Finv, Cinv, dμ_dθ, score, dC_dx, 1.)],
                axis=1)
        print("Calculate regulariser")
        dreg_dx, reg, r = self.get_regularisation_derivative(C, Cinv, dC_dx)
        print("Build gradients")
        gradients = []
        for layer in range(len(self.model.variables)):
            gradients.append(
                tf.divide(
                    tf.einsum(
                        "ij,ij...->...",
                        dΛ_dx,
                        dx_dw[layer]),
                    tf.dtypes.cast(
                        self.n_s,
                        self.dtype)))
            if self.numerical:
                gradients[layer] = tf.add(
                    gradients[layer],
                    tf.divide(
                        tf.einsum(
                            "ijkl,ijkl...->...",
                            ddΛ_dxdθ,
                            ddx_dwdθ[layer]),
                        tf.dtypes.cast(
                            self.n_d,
                            self.dtype)))
        print("Apply gradients")
        self.optimiser.apply_gradients(zip(gradients, self.model.variables))
        print("Return diagnostics")
        return (tf.linalg.det(F), tf.linalg.det(C), tf.linalg.det(Cinv),
                dμ_dθ, reg, r)

    @tf.function
    def numerical_validate(self, data_iterator, derivative_iterator=None):
        if self.single_dataset:
            if self.use_summaries:
                d, dd_dθ, s, ds_dθ = next(data_iterator)
            else:
                d, dd_dθ = next(data_iterator)
        else:
            if self.use_summaries:
                d, s = next(data_iterator)
                dd_dθ, ds_dθ = next(derivative_iterator)
            else:
                d = next(data_iterator)
                dd_dθ = next(derivative_iterator)
        x = self.model(d)
        dx_dθ = self.model(dd_dθ)
        C, Δμ = self.get_covariance(x)
        Cinv = tf.linalg.inv(C)
        dμ_dθ = self.get_numerical_derivative_mean(dx_dθ, self.test_δθ)
        score = self.get_score(Cinv, dμ_dθ)
        F = self.get_fisher(Cinv, dμ_dθ, score)
        reg, _ = self.get_regularisation(C, Cinv)
        r, _ = self.get_r(reg)
        return (tf.linalg.det(F), tf.linalg.det(C), tf.linalg.det(Cinv), dμ_dθ,
                reg, r)

    def get_covariance(self, x):
        μ = tf.reduce_mean(
            x,
            axis=0,
            keepdims=True,
            name="mean")
        Δμ = tf.subtract(
            x,
            μ,
            name="centred_mean")
        C = tf.divide(
            tf.einsum(
                "ij,ik->jk",
                Δμ,
                Δμ,
                name="unnormalised_covariance"),
            self.n_sm1,
            name="covariance")
        return C, Δμ

    #def get_chain_derivative_mean(self, dx_dd, dd_dθ):
    #    return tf.einsum("i,i->")

    def get_numerical_derivative_mean(self, dx, δθ):
        return tf.reduce_mean(
            tf.multiply(
                tf.subtract(
                    dx[:, 1, ...],
                    dx[:, 0, ...]),
                tf.expand_dims(tf.expand_dims(δθ, 0), 2)),
            axis=0,
            name="numerical_derivative_mean_wrt_parameters")

    def get_score(self, Cinv, dμ_dθ):
        return tf.einsum(
            "ij,kj->ki",
            Cinv,
            dμ_dθ,
            name="score")

    def get_fisher(self, Cinv, dμ_dθ, score):
        F = tf.linalg.band_part(
            tf.einsum(
                "ij,kj->ik",
                dμ_dθ,
                score,
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

    def get_covariance_derivative(self, Δμ):
        return tf.divide(
            tf.reduce_sum(
                tf.add(
                    tf.einsum(
                        "ijkl,im->ijkml",
                        self.dΔμ_dx,
                        Δμ,
                        name="centred_mean_derivative_first_half"),
                    tf.einsum(
                        "ij,iklm->ikjlm",
                        Δμ,
                        self.dΔμ_dx,
                        name="centred_mean_derivative_second_half"),
                    name="centred_mean_derivative"),
                axis=0,
                name="unnormalised_covariance_derivative"),
            self.n_sm1,
            name="covariance_derivative")

    def get_fisher_derivative(self, Cinv, dμdθ, score, dCdx):
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
                        score,
                        name="covariance_derivative_score"),
                    name="inverse_covariance_covariance_derivative_score"),
                name="half_derivative_of_fisher_wrt_summaries"),
            0,
            -1,
            name="triangle_derivative_of_fisher_wrt_summaries")
        return tf.multiply(
            -0.5,
            tf.add(
                dFdx,
                tf.transpose(
                    dFdx,
                    perm=[0, 1, 3, 2],
                    name="transposed_derivative_of_fisher_wrt_summaries"),
                name="double_derivative_of_fisher_wrt_summaries"),
            name="derivative_of_fisher_wrt_summaries")

    def get_numerical_fisher_derivative(self, Cinv, dμdθ, score, sign):
        dFdx_a = tf.linalg.band_part(
            tf.einsum(
                "ijklm,nk->imljn",
                tf.multiply(sign, self.d2μ_dθdx),
                score), 0, -1)
        dFdx_b = tf.linalg.band_part(
            tf.einsum(
                "ij,kljmn->knmil",
                dμdθ,
                tf.einsum(
                    "ij,kljmn->klimn",
                    Cinv,
                    tf.multiply(sign, self.d2μ_dθdx))), 0, -1)
        return tf.multiply(
            0.5,
            tf.add(
                tf.add(
                    dFdx_a,
                    tf.transpose(
                        dFdx_a,
                        perm=[0, 1, 2, 4, 3])),
                tf.add(
                    dFdx_b,
                    tf.transpose(
                        dFdx_b,
                        perm=[0, 1, 2, 4, 3]))))

    def get_loss(self, Finv, Cinv, dμ_dθ, score, dC_dx):
        dF_dx = self.get_fisher_derivative(Cinv, dμ_dθ, score, dC_dx)
        return -tf.linalg.trace(
            tf.einsum(
                "ij,kljm->klim",
                Finv,
                dF_dx),
            name="derivative_of_logdetfisher_wrt_summaries")

    def get_num_loss(self, Finv, Cinv, dμ_dθ, score, dC_dx, sign):
        dF_dx = self.get_numerical_fisher_derivative(Cinv, dμ_dθ, score, sign)
        return -tf.linalg.trace(
            tf.einsum(
                "ij,klmjn->klmin",
                Finv,
                dF_dx),
            name="derivative_of_logdetfisher_wrt_lower_summaries")

    def get_regularisation(self, C, Cinv):
        CmI = tf.subtract(C, self.identity)
        CinvmI = tf.subtract(Cinv, self.identity)
        regulariser = tf.multiply(
            0.5,
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
        return regulariser, CmI

    def get_r(self, regulariser):
        rate = tf.multiply(-self.α, regulariser)
        e_rate = tf.exp(rate)
        r = tf.divide(
                tf.multiply(
                    self.λ,
                    regulariser),
                tf.add(
                    regulariser,
                    e_rate))
        return r, e_rate

    def get_regularisation_strength(self, regulariser):
        r, e_rate = self.get_r(regulariser)
        dr_dregulariser = tf.multiply(
            r,
            tf.add(
                1.,
                tf.divide(
                    tf.multiply(
                        tf.add(
                            1.,
                            tf.multiply(
                                self.α,
                                regulariser)),
                        e_rate),
                    tf.add(
                        regulariser,
                        e_rate))))
        return r, dr_dregulariser

    def get_regularisation_derivative(self, C, Cinv, dC_dx):
        reg, CmI = self.get_regularisation(C, Cinv)
        r, dr_dreg = self.get_regularisation_strength(reg)
        Cinv2 = tf.einsum(
            "ij,jk->ik",
            Cinv,
            Cinv)
        Cinv3 = tf.einsum(
            "ij,jk->ik",
            Cinv2,
            Cinv)
        dreg_dx = tf.multiply(
            tf.add(
                tf.multiply(
                    reg,
                    dr_dreg),
                r),
            tf.linalg.trace(
                tf.einsum(
                    "ij,kjlm->kilm",
                    tf.add(
                        CmI,
                        tf.subtract(
                            Cinv2,
                            Cinv3)),
                    dC_dx)))
        return dreg_dx, reg, r

    def set_regularisation_strength(self, ϵ, λ):
        self.λ = tf.Variable(λ, dtype=self.dtype, trainable=False,
                             name="strength")
        self.α = -tf.divide(
            tf.math.log(
                tf.add(tf.multiply(tf.subtract(λ, 1.), ϵ),
                       tf.divide(tf.square(ϵ), tf.add(1., ϵ)))),
            ϵ)
