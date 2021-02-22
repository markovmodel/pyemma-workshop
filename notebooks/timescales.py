def implied_timescales_msm(dtrajs, lagtimes, nits=None, bayesian: bool = True):
    import deeptime as dt
    from tqdm.autonotebook import tqdm

    count_mode = 'effective' if bayesian else 'sliding'
    its = []
    its_stats = []
    for lag in tqdm(lagtimes, leave=False, desc="Computing implied timescales"):
        counts = dt.markov.TransitionCountEstimator(lag, count_mode).fit(dtrajs).fetch_model().submodel_largest()
        if bayesian:
            bmsm = dt.markov.msm.BayesianMSM().fit(counts).fetch_model()
            its.append(bmsm.prior.timescales(k=nits))
            its_stats.append(bmsm.evaluate_samples('timescales', k=nits).T)
        else:
            msm = dt.markov.msm.MaximumLikelihoodMSM().fit(counts).fetch_model()
            its.append(msm.timescales(k=nits))
    if bayesian:
        return lagtimes, its, its_stats
    else:
        return lagtimes, its


def implied_timescales_hmm(dtrajs, lagtimes, n_hidden_states, nits=None, bayesian: bool = True):
    import deeptime as dt
    from tqdm.autonotebook import tqdm

    its = []
    its_stats = []

    for lag in tqdm(lagtimes, leave=False, desc="Computing implied timescales"):
        if bayesian:
            bhmm_estimator = dt.markov.hmm.BayesianHMM.default(dtrajs, n_hidden_states, lag)
            bhmm_estimator.n_samples = 50
            bhmm = bhmm_estimator.fit(dtrajs).fetch_model()
            its.append(bhmm.prior.transition_model.timescales(k=nits))
            its_stats.append(bhmm.evaluate_samples('transition_model/timescales', k=nits).T)
        else:
            hmm_init = dt.markov.hmm.init.discrete.metastable_from_data(dtrajs, n_hidden_states, lagtime=lag)
            hmm = dt.markov.hmm.MaximumLikelihoodHMM(hmm_init).fit(dtrajs).fetch_model()
            its.append(hmm.transition_model.timescales(k=nits))
    if bayesian:
        return lagtimes, its, its_stats
    else:
        return lagtimes, its
