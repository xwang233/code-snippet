import matplotlib.pyplot as plt
import seaborn as sns
import json
import collections

THRESHOLD = 30000

d = {
    "1.8": "res-56b43f4",
    "1.9": "res-gitbf16699"
}

j = {}

for k, v in d.items():
    with open(f'{v}.json', 'r') as f:
        j[k] = json.load(f)


keys = j["1.8"].keys()
# print(keys)

single_matrix = collections.defaultdict(list)
batched_matrix = collections.defaultdict(list)

for s_key in keys:
    key = tuple(json.loads(s_key))
    # print(key)

    if key[0] == 1:
        single_matrix[key].append(j['1.8'][s_key])
        single_matrix[key].append(j['1.9'][s_key])
    else:
        batched_matrix[key].append(j['1.8'][s_key])
        batched_matrix[key].append(j['1.9'][s_key])

# print(single_matrix)

sx18 = [k[1] for k in single_matrix if j['1.8'][str(list(k))] < THRESHOLD]
sy18 = [j['1.8'][str(list(k))] for k in single_matrix if j['1.8'][str(list(k))] < THRESHOLD]

sx19 = [k[1] for k in single_matrix]
sy19 = [j['1.9'][str(list(k))] for k in single_matrix]

plt.plot(sx18, sy18)
plt.plot(sx19, sy19)
plt.xlim(min(sx19) - 10, max(sx19) + 10)
plt.legend(['1.8', '1.9'])
plt.title(f'Cholesky decomposition on single matrix (GPU)')
plt.xlabel('matrix_size')
plt.ylabel(u'execution_time (μs)')
plt.savefig('single.png', transparent=True)

plt.clf()

for mat_size in (4, 32, 128):
    plt.clf()

    # THRESHOLD = 100
    ver = '1.8'
    sx18_b = [k[0] for k in batched_matrix if j[ver][str(list(k))] < THRESHOLD and k[1] == mat_size]
    sy18_b = [j[ver][str(list(k))] for k in batched_matrix if j[ver][str(list(k))] < THRESHOLD and k[1] == mat_size]

    # THRESHOLD = 63
    ver = '1.9'
    sx19_b = [k[0] for k in batched_matrix if j[ver][str(list(k))] < THRESHOLD and k[1] == mat_size]
    sy19_b = [j[ver][str(list(k))] for k in batched_matrix if j[ver][str(list(k))] < THRESHOLD and k[1] == mat_size]

    plt.plot(sx18_b, sy18_b)
    plt.plot(sx19_b, sy19_b)
    plt.xlim(min(sx19_b) - 10, max(sx19_b) + 10)
    plt.legend(['1.8', '1.9'])
    plt.title(f'Cholesky decomposition on batched matrices of {mat_size}x{mat_size} (GPU)')
    plt.xlabel('batch_size')
    plt.ylabel(u'execution_time (μs)')
    plt.savefig(f'batch_{mat_size}.png', transparent=True)

    plt.clf()