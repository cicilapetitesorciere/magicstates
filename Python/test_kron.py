from definitions import kron, x, y, z, one, plusstate
from mpmath import mp

n_repetitions: int

for n_repetitions in []:
    output = kron(*([one] * n_repetitions))

    for i in range(output.rows):
        for j in range(output.cols):
            assert(output[i,j] == (1.0 if i == j else 0.0))

for n_repetitions in []:
    output = kron(*([x] * n_repetitions))

    # print(output)

    for i in range(output.rows):
        for j in range(output.cols):
            assert(output[i, output.cols - 1 - j] == (1.0 if i == j else 0.0))


for n_repetitions in []:
    output = kron(*([z] * n_repetitions))

    print(output)


    for i in range(output.rows):
        for j in range(output.cols):
            if i == j:
                correct_value = 1.0
                b = 1
                while True:
                    if j & b != 0:
                        correct_value *= -1.0
                    b *= 2
                    if b > i:
                        break

                assert(output[i,i] == correct_value)
                
            else:
                assert(output[i, j] == 0)

for n_repetitions in []:
    output = kron(*([z] * n_repetitions))

    print(output)


    for i in range(output.rows):
        for j in range(output.cols):
            if i == j:
                correct_value = 1.0
                b = 1
                while True:
                    if j & b != 0:
                        correct_value *= -1.0
                    b *= 2
                    if b > i:
                        break

                assert(output[i,i] == correct_value)
                
            else:
                assert(output[i, j] == 0)


for n_repetitions in []:
    output = kron(*([plusstate] * n_repetitions))

    for i in range(output.rows):
        for j in range(output.cols):
            assert(output[i,j] == (1.0 if i == j else 0.0))

print("Success!")