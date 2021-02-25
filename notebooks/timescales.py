import multiprocessing as _mp
import deeptime as _dt


def _worker_its_msm(args):
    i, dtrajs, lag, nits, bayesian = args
    count_mode = 'effective' if bayesian else 'sliding'
    its, its_stats = None, None

    counts = _dt.markov.TransitionCountEstimator(lag, count_mode).fit(dtrajs, n_jobs=1).fetch_model().submodel_largest()
    if bayesian:
        bmsm = _dt.markov.msm.BayesianMSM(n_samples=50).fit(counts).fetch_model()
        its = bmsm.prior.timescales(k=nits)
        its_stats = bmsm.evaluate_samples('timescales', k=nits).T
    else:
        msm = _dt.markov.msm.MaximumLikelihoodMSM().fit(counts).fetch_model()
        its = msm.timescales(k=nits)
    return i, its, its_stats


def implied_timescales_msm(dtrajs, lagtimes, nits=None, bayesian: bool = True, n_jobs=None):
    from deeptime.util.parallel import joining, handle_n_jobs
    from tqdm.autonotebook import tqdm

    n_jobs = handle_n_jobs(n_jobs)

    its = [None for _ in range(len(lagtimes))]
    its_stats = [None for _ in range(len(lagtimes))]

    args = [(i, dtrajs, lagtimes[i], nits, bayesian) for i in range(len(lagtimes))]

    if n_jobs > 1:
        with joining(_mp.get_context("spawn").Pool(processes=n_jobs)) as pool:
            for result in tqdm(pool.imap_unordered(_worker_its_msm, args), leave=False,
                               desc="Computing implied timescales", total=len(args)):
                i, its_, its_samples_ = result
                its[i] = its_
                its_stats[i] = its_samples_
    else:
        for arg in tqdm(args, leave=False, desc="Computing implied timescales"):
            i, its_, its_samples_ = _worker_its_msm(arg)
            its[i] = its_
            its_stats[i] = its_samples_
    if bayesian:
        return lagtimes, its, its_stats
    else:
        return lagtimes, its


def _worker_its_hmm(args):
    i, dtrajs, lag, n_hidden_states, nits, bayesian = args

    its = None
    its_stats = None

    if bayesian:
        bhmm_estimator = _dt.markov.hmm.BayesianHMM.default(dtrajs, n_hidden_states, lag)
        bhmm_estimator.n_samples = 50
        bhmm = bhmm_estimator.fit(dtrajs).fetch_model()
        its = bhmm.prior.transition_model.timescales(k=nits)
        its_stats = bhmm.evaluate_samples('transition_model/timescales', k=nits).T
    else:
        hmm_init = _dt.markov.hmm.init.discrete.metastable_from_data(dtrajs, n_hidden_states, lagtime=lag)
        hmm = _dt.markov.hmm.MaximumLikelihoodHMM(hmm_init).fit(dtrajs).fetch_model()
        its = hmm.transition_model.timescales(k=nits)

    return i, its, its_stats


def implied_timescales_hmm(dtrajs, lagtimes, n_hidden_states, nits=None, bayesian: bool = True, n_jobs=None):
    from deeptime.util.parallel import joining, handle_n_jobs
    from tqdm.autonotebook import tqdm

    n_jobs = handle_n_jobs(n_jobs)
    its = [None for _ in range(len(lagtimes))]
    its_stats = [None for _ in range(len(lagtimes))]

    args = [(i, dtrajs, lagtimes[i], n_hidden_states, nits, bayesian) for i in range(len(lagtimes))]

    if n_jobs > 1:
        with joining(_mp.get_context("spawn").Pool(processes=n_jobs)) as pool:
            for result in tqdm(pool.imap_unordered(_worker_its_hmm, args), leave=False,
                               desc="Computing implied timescales", total=len(args)):
                i, its_, its_samples_ = result
                its[i] = its_
                its_stats[i] = its_samples_
    else:
        for arg in tqdm(args, leave=False, desc="Computing implied timescales"):
            i, its_, its_samples_ = _worker_its_hmm(arg)
            its[i] = its_
            its_stats[i] = its_samples_
    if bayesian:
        return lagtimes, its, its_stats
    else:
        return lagtimes, its
