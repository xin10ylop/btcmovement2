"""EDA on Jan-Apr 2025: is the short-horizon drift an 'efficient-price gap' shared across horizons?"""
import glob, numpy as np
files = sorted(glob.glob('/home/user/data/days/2025-0[1-4]-*.npz'))
cols = ['s_close','s_volume','s_tb_base','p_close','p_buy','p_sell']
D = {c: np.concatenate([np.load(f)[c] for f in files]) for c in cols}
pc = D['p_close']; m = np.isnan(pc); idx = np.where(~m, np.arange(len(pc)), 0); np.maximum.accumulate(idx, out=idx); pc = pc[idx]
ls = np.log(D['s_close']); lp = np.log(pc); N = len(ls)
rs = np.diff(ls, prepend=ls[0]); rp = np.diff(lp, prepend=lp[0])
b = lp - ls
cs = lambda a: np.cumsum(np.r_[0.0, a])
def past_sum(a, L):   # sum over (t-L, t]
    c = cs(a); out = np.full(N, np.nan); out[L-1:] = c[L:] - c[:-L]; return out
def lag_diff(x, L):
    out = np.full(N, np.nan); out[L:] = x[L:] - x[:-L]; return out
# local vol scale: trailing 300s rms of 1s return
v300 = np.sqrt(past_sum(rs**2, 300) / 300); v300 = np.maximum(v300, np.nanmedian(v300)*0.25)
X = {}
X['bdev60'] = (b - past_sum(b, 60)/60)/v300
X['bdev600'] = (b - past_sum(b, 600)/600)/v300
for L in [1, 2, 5, 15, 60, 300]:
    X[f'rs{L}'] = lag_diff(ls, L)/v300
    X[f'rp{L}'] = lag_diff(lp, L)/v300
sb = D['s_tb_base']; ssell = D['s_volume'] - sb
for L in [5, 30, 300]:
    tot = past_sum(sb+ssell, L) + 1e-9
    X[f'simb{L}'] = (past_sum(sb, L) - past_sum(ssell, L))/tot
    totp = past_sum(D['p_buy']+D['p_sell'], L) + 1e-9
    X[f'pimb{L}'] = (past_sum(D['p_buy'], L) - past_sum(D['p_sell'], L))/totp
names = list(X)
F = np.clip(np.column_stack([X[k] for k in names]), -10, 10)
t = np.arange(N)
samp = t[(t > 4000) & (t < N - 400)][::7]       # sparse sample
half = len(samp)//2
for H in [5, 30, 60, 300]:
    y = np.clip((np.r_[ls[H:] - ls[:-H], np.full(H, np.nan)])[samp] / v300[samp], -30*np.sqrt(H), 30*np.sqrt(H))
    Xs = F[samp]
    ok = np.isfinite(y) & np.isfinite(Xs).all(1)
    Xs, y = Xs[ok], y[ok]
    tr = np.arange(len(y)) < half*0.98; te = np.arange(len(y)) > half*1.02   # Jan-Feb fit, Mar-Apr test
    A = np.c_[np.ones(tr.sum()), Xs[tr]]
    beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
    pred = np.c_[np.ones(te.sum()), Xs[te]] @ beta
    r2 = 1 - ((y[te]-pred)**2).mean()/((y[te]-y[tr].mean())**2).mean()
    cor = np.corrcoef(pred, y[te])[0,1]
    print(f'H={H:4d}  OOS R2={r2*100:.3f}%  corr={cor:.4f}  sd(pred)={pred.std():.4f} (y in units of 1s-vol; sd(y)={y[te].std():.2f})')
    top = np.argsort(-np.abs(beta[1:]))[:6]
    print('    top coefs:', ', '.join(f'{names[i]}={beta[1+i]:+.3f}' for i in top))
