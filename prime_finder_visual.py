import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpmath import li as li_n

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def simple_sieve(limit):
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i:limit+1:i] = False
    return np.nonzero(sieve)[0]

def segmented_sieve(limit, segment_size=10**6):
    base_primes = simple_sieve(int(limit**0.5) + 1)
    primes = []

    for low in tqdm(range(2, limit + 1, segment_size), desc="Segmenting"):
        high = min(low + segment_size - 1, limit)
        sieve = np.ones(high - low + 1, dtype=bool)

        for p in base_primes:
            start = max(p*p, ((low + p - 1) // p) * p)
            sieve[start - low : high - low + 1 : p] = False

        for i, is_prime in enumerate(sieve):
            if is_prime:
                primes.append(low + i)

    return np.array(primes, dtype=np.int64)


def gpu_sieve(n):
    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy not installed or GPU unavailable.")
    sieve = cp.ones(n + 1, dtype=cp.bool_)
    sieve[:2] = False
    for i in range(2, int(cp.sqrt(n).item()) + 1):
        if sieve[i]:
            sieve[i*i : n+1 : i] = False
    return cp.nonzero(sieve)[0].get()


def save_primes(primes, filename="primes.npy"):
    np.save(filename, primes)
    print(f"✅ Saved {len(primes):,} primes to {filename}")

def load_primes(filename="primes.npy"):
    primes = np.load(filename)
    print(f"✅ Loaded {len(primes):,} primes from {filename}")
    return primes

def plot_prime_density(primes, max_n=None, bins=1000):
    if max_n is None:
        max_n = primes[-1]

    plt.figure(figsize=(10, 6))
    plt.hist(primes, bins=bins, range=(0, max_n), color='royalblue', alpha=0.7)
    plt.title(f"Prime Distribution up to {max_n:,}")
    plt.xlabel("Number")
    plt.ylabel("Prime Count")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("prime_density.png")
    plt.show()

def plot_prime_gaps(primes):
    gaps = np.diff(primes)
    plt.figure(figsize=(10, 5))
    plt.plot(primes[:-1], gaps, lw=0.4, color='darkorange')
    plt.title("Prime Gaps")
    plt.xlabel("Prime")
    plt.ylabel("Gap to Next Prime")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("prime_gaps.png")
    plt.show()

def find_primes(limit=50_000_000, use_gpu=False, save=True):
    print(f"{'🔧 Using GPU' if use_gpu else '🧠 Using CPU'} to find primes up to {limit:,}")
    start = time.time()
    if use_gpu and GPU_AVAILABLE:
        primes = gpu_sieve(limit)
    else:
        primes = segmented_sieve(limit)
    elapsed = time.time() - start
    print(f"✅ Found {len(primes):,} primes in {elapsed:.2f} seconds.")
    
    if save:
        save_primes(primes, f"primes_up_to_{limit}.npy")
    
    return primes

def plot_prime_count_comparison(primes, max_n):
    import matplotlib.pyplot as plt
    from math import log
    from mpmath import li as li_n

    step = max_n // 1000
    if step == 0: step = 1
    xs = np.arange(10, max_n, step)
    pi_actual = np.searchsorted(primes, xs, side='right')
    pi_log = xs / np.log(xs)
    pi_li = np.array([float(li_n(x)) for x in xs])

    # π(n) Comparison Plot
    plt.figure(figsize=(12, 6))
    plt.plot(xs, pi_actual, label="π(n): Actual prime count", lw=2, color="navy")
    plt.plot(xs, pi_log, label="n / log(n): Approx. by Prime Number Theorem", lw=1.5, linestyle='--', color="orange")
    plt.plot(xs, pi_li, label="Li(n): Logarithmic Integral Estimate", lw=1.5, linestyle=':', color="green")

    plt.title("Prime Counting Function π(n) vs Approximations")
    plt.xlabel("n (Number Range)")
    plt.ylabel("Number of Primes ≤ n")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("pi_comparison.png")
    plt.show()

    # Error Plot
    plt.figure(figsize=(12, 5))
    plt.plot(xs, pi_log - pi_actual, label="n/log(n) − π(n)", color='red', lw=1)
    plt.plot(xs, pi_li - pi_actual, label="Li(n) − π(n)", color='green', lw=1)

    plt.title("Approximation Error: π(n) vs Estimates")
    plt.xlabel("n (Number Range)")
    plt.ylabel("Difference in Prime Counts")
    plt.axhline(0, color='black', lw=0.5)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("pi_error.png")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prime Finder with Visualization")
    parser.add_argument("--limit", type=int, default=50_000_000, help="Upper bound for primes")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration (CuPy)")
    parser.add_argument("--load", type=str, help="Load primes from .npy file")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument("--plot", action="store_true", help="Plot prime density and gaps")

    args = parser.parse_args()

    if args.load:
        primes = load_primes(args.load)
    else:
        primes = find_primes(limit=args.limit, use_gpu=args.gpu, save=not args.no_save)

    if args.plot:
        plot_prime_gaps(primes)
        plot_prime_density(primes, max_n=args.limit)
        plot_prime_count_comparison(primes, max_n=args.limit)


