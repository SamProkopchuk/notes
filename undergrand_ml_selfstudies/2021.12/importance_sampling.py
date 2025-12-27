import random


def E(f):
    # Return E_f(X)
    return sum(i * f[i] for i in range(len(f)))


def importance_sample(f, g, k=1_000_000):
    '''
    Uses importance sampling to calculate E_f(X) by sampling from X ~ g
    '''
    samples = random.choices(range(len(g)), weights=g, k=k)
    return sum(x * f[x] / g[x] for x in samples) / k


def main(size=6):
    # Define two pdf f and g where f[i] == probability x == i
    f = [1 / size] * size
    g = [random.random() for _ in range(size)]
    # Normalize g:
    g = [v / sum(g) for v in g]

    print(f'True value of E_f(X): {E(f)}')
    print(f'True value of E_g(X): {E(g)}')
    print(f'Value of E_f(X) via I.S.: {importance_sample(f, g)}')
    print(f'Value of E_g(X) via I.S.: {importance_sample(g, f)}')


if __name__ == '__main__':
    main()
