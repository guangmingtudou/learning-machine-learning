def print_V(V, nrow, ncol):
    for i in range(nrow):
        for j in range(ncol):
            print("%.6f" % V[i*ncol+j], end=' ')
        print(end='\n')
    print()

def print_P(P, n):
    for i in range(n):
        for j in range(n):
            print("%.1f" % P[i][j], end=' ')
        print(end='\n')
    print()