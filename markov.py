import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import threading

###############################################################################
# PUBLIC COINBASE DATA FETCHER (NO API KEY)
###############################################################################

def fetch_public_candles(product="BTC-USD", granularity=3600, limit=300):
    """
    Fetch historical candles from Coinbase's public unauthenticated API.
    granularity in seconds: 60, 300, 900, 3600, 21600, 86400
    """
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    params = {"granularity": granularity}
    
    r = requests.get(url, params=params)
    r.raise_for_status()
    raw = r.json()

    # Coinbase returns: [ [time, low, high, open, close, volume], ... ]
    df = pd.DataFrame(raw, columns=["time","low","high","open","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time")
    df.reset_index(drop=True, inplace=True)
    return df


###############################################################################
# MINIMAL GAUSSIAN HMM (the same structure used earlier)
###############################################################################
def logsumexp(a, axis=None):
    a = np.asarray(a)
    m = np.max(a, axis=axis, keepdims=True)
    s = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    
    # If axis is None → return scalar
    if axis is None:
        return float(s)
    
    # Otherwise → return a 1-D array
    return np.squeeze(s, axis=axis)



class GaussianHMM:
    def __init__(self, n_states=3, cov_type="diag", n_iter=50, tol=1e-4):
        self.K = n_states
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.tol = tol

    def _init_params(self, X):
        T, D = X.shape
        self.D = D
        # init mixing
        self.pi = np.ones(self.K) / self.K
        self.A = np.ones((self.K, self.K))
        self.A /= self.A.sum(axis=1, keepdims=True)

        # init Gaussian means/covs
        idx = np.random.choice(T, self.K, replace=False)
        self.means = X[idx] + 1e-4*np.random.randn(self.K, D)

        if self.cov_type == "diag":
            self.covs = np.array([np.var(X, axis=0) + 1e-3 for _ in range(self.K)])
        else:
            base = np.cov(X.T) + 1e-3*np.eye(D)
            self.covs = np.array([base for _ in range(self.K)])

    def _log_gauss(self, X):
        T = len(X)
        logB = np.zeros((T, self.K))
        for k in range(self.K):
            diff = X - self.means[k]
            if self.cov_type == "diag":
                var = self.covs[k]
                ll = -0.5 * (np.sum(diff*diff/var, axis=1)
                             + np.sum(np.log(2*np.pi*var)))
            else:
                cov = self.covs[k]
                inv = np.linalg.inv(cov)
                det = np.linalg.det(cov)
                ll = -0.5 * (np.sum(diff @ inv * diff, axis=1)
                             + np.log(det) + self.D*np.log(2*np.pi))
            logB[:,k] = ll
        return logB

    def _forward_backward(self, logB):
        T, K = logB.shape
        log_pi = np.log(self.pi + 1e-12)
        logA = np.log(self.A + 1e-12)

        # forward
        log_alpha = np.zeros((T,K))
        log_alpha[0] = log_pi + logB[0]
        for t in range(1,T):
            temp = log_alpha[t-1][:,None] + logA
            log_alpha[t] = logsumexp(temp, axis=0)[0] + logB[t]

        # backward
        log_beta = np.zeros((T,K))
        for t in reversed(range(T-1)):
            temp = logA + logB[t+1] + log_beta[t+1]
            log_beta[t] = logsumexp(temp, axis=1)

        # posteriors
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1)[:, None]

        gamma = np.exp(log_gamma)

        # pairwise posteriors
        xi_sum = np.zeros((K,K))
        for t in range(T-1):
            temp = (
                log_alpha[t][:,None]
                + logA
                + logB[t+1][None,:]
                + log_beta[t+1][None,:]
            )
            temp -= logsumexp(temp)
            xi_sum += np.exp(temp)

        return gamma, xi_sum

    def fit(self, X):
        self._init_params(X)
        prev_ll = -np.inf

        for i in range(self.n_iter):
            logB = self._log_gauss(X)
            gamma, xi_sum = self._forward_backward(logB)

            ll = np.sum(logsumexp(logB, axis=1))
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

            # M-step
            self.pi = gamma[0]
            self.A = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-12)

            # means
            self.means = (gamma.T @ X) / gamma.sum(axis=0)[:,None]

            # covs
            if self.cov_type == "diag":
                covs = np.zeros((self.K, self.D))
                for k in range(self.K):
                    diff = X - self.means[k]
                    covs[k] = (gamma[:,k][:,None]*(diff*diff)).sum(axis=0) / gamma[:,k].sum()
                    covs[k] += 1e-6
                self.covs = covs
            else:
                covs = np.zeros((self.K, self.D, self.D))
                for k in range(self.K):
                    diff = X - self.means[k]
                    covs[k] = (diff.T * gamma[:,k]) @ diff
                    covs[k] /= gamma[:,k].sum()
                    covs[k] += 1e-6*np.eye(self.D)
                self.covs = covs

    def predict_state(self, X):
        logB = self._log_gauss(X)
        gamma, _ = self._forward_backward(logB)
        return np.argmax(gamma[-1])       # last time step


###############################################################################
# LIVE LOOP — fetch public Coinbase data every N seconds and run HMM
###############################################################################

def live_hmm_loop(product="BTC-USD", granularity=3600, interval=60):
    """
    Fetch Coinbase public data every `interval` seconds and rerun HMM on it.
    """
    hmm = GaussianHMM(n_states=3, cov_type="diag", n_iter=25)

    while True:
        print("\n=== Fetching Coinbase Public Data ===")
        df = fetch_public_candles(product, granularity)
        close = df["close"].astype(float).values

        # Features
        logret = np.diff(np.log(close), prepend=close[0])
        vol = pd.Series(logret).rolling(20).std().bfill().values
        X = np.column_stack([logret, vol])

        # Train HMM
        hmm.fit(X)

        # Predict current state
        state = hmm.predict_state(X)
        state_mean = hmm.means[state, 0]

        signal = "LONG" if state_mean > 0 else "FLAT"

        print(f"Time: {datetime.utcnow()} | State: {state} | Mean Return: {state_mean:.6f} | Signal: {signal}")

        time.sleep(interval)


###############################################################################
# RUN LIVE LOOP
###############################################################################

if __name__ == "__main__":
    # Update every 60 seconds using 1-hour candles
    live_hmm_loop(
        product="BTC-USD",
        granularity=3600,   # 1-hour
        interval=60         # refresh every 1 minute
    )
