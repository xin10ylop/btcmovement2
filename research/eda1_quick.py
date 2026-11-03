"""EDA on Jan-Apr 2025 (training period only): basic microstructure facts."""
import glob, numpy as np, os
from datetime import date
files = sorted(glob.glob('/home/user/data/days/2025-0[1-4]-*.npz'))
print(len(files), 'days')
cols = ['s_close','s_open','s_high','s_low','s_volume','s_tb_base','s_count','p_close','p_buy','p_sell','p_ntr']
D = {c: [] for c in cols}
for f in files:
    z = np.load(f)
    for c in cols: D[c].append(z[c])
D = {c: np.concatenate(v) for c, v in D.items()}
pc = D['p_close'].copy()
m = np.isnan(pc); idx = np.where(~m, np.arange(len(pc)), 0); np.maximum.accumulate(idx, out=idx); pc = pc[idx]
ls = np.log(D['s_close']); lp = np.log(pc)
rs = np.diff(ls, prepend=ls[0]); rp = np.diff(lp, prepend=lp[0])
N = len(ls)
print('1s spot ret std (bps)', rs.std()*1e4, ' perp', rp.std()*1e4)
print('annualized vol from 1s', rs.std()*np.sqrt(365*86400))
# zero-return fraction
print('frac zero 1s spot returns', (rs==0).mean(), ' zero-trade seconds', (D['s_count']==0).mean())
# autocorrelation of 1s spot returns
for k in [1,2,3,5,10]:
    print(f'acf spot lag{k}: {np.corrcoef(rs[k:], rs[:-k])[0,1]:.4f}   perp->spot lead lag{k}: {np.corrcoef(rs[k:], rp[:-k])[0,1]:.4f}  spot->perp: {np.corrcoef(rp[k:], rs[:-k])[0,1]:.4f}')
print('contemporaneous corr spot/perp 1s', np.corrcoef(rs, rp)[0,1])
# basis
b = (lp - ls)*1e4
print('basis bps mean/std', b.mean(), b.std())
# basis deviation from 60s mean predicts future spot return?
def roll_mean(a, L):
    c = np.cumsum(np.r_[0, a]); out = np.full(len(a), np.nan); out[L-1:] = (c[L:] - c[:-L]) / L; return out
for L in [10, 60, 300]:
    dev = b - roll_mean(b, L)
    for H in [1, 5, 30]:
        fut = np.r_[ls[H:] - ls[:-H], np.full(H, np.nan)]
        ok = ~np.isnan(dev) & ~np.isnan(fut)
        print(f'basis dev(L={L}) vs future spot ret H={H}: corr {np.corrcoef(dev[ok], fut[ok])[0,1]:.4f}')
# 5-minute windows: ties and up freq
nw = N // 300
o = D['s_open'][::300][:nw]; c = D['s_close'][299::300][:nw]
print('windows', nw, ' P(close>open)', (c>o).mean(), ' P(close==open)', (c==o).mean())
r5 = np.log(c/o)
print('5m ret std bps', r5.std()*1e4, ' kurtosis', ((r5-r5.mean())**4).mean()/r5.var()**2)
# seasonality by second of minute and minute of hour
sec_of_day = np.arange(N) % 86400
a = np.abs(rs)
som = np.bincount(sec_of_day % 60, weights=a, minlength=60) / np.bincount(sec_of_day % 60, minlength=60)
print('mean |r| by second-of-minute (rel):', np.round(som / som.mean(), 2))
soh = np.bincount(sec_of_day % 3600 // 60, weights=a, minlength=60) / np.bincount(sec_of_day % 3600 // 60, minlength=60)
print('mean |r| by minute-of-hour (rel):', np.round(soh / soh.mean(), 2))
hod = np.bincount(sec_of_day // 3600, weights=a, minlength=24) / np.bincount(sec_of_day // 3600, minlength=24)
print('mean |r| by hour-of-day UTC (rel):', np.round(hod / hod.mean(), 2))
# within 5m window: second-of-window
sow = np.bincount(sec_of_day % 300, weights=a, minlength=300) / np.bincount(sec_of_day % 300, minlength=300)
print('second-of-window first 10 / last 10 (rel):', np.round(sow[:10]/sow.mean(),2), np.round(sow[-10:]/sow.mean(),2))
