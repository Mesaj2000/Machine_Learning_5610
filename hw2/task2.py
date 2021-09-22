from math import log

def entropy(c):
	a, b = c
	ni = a + b
	ent_a = 0 if a == 0 else (a/ni) * log(a/ni, 2)
	ent_b = 0 if b == 0 else (b/ni) * log(b/ni, 2)
	ent = - (ent_a + ent_b)
	return ent

def combine_entropies(m1, n1, m2, n2, n):
    return (n1 * m1 + n2 * m2) / n

def task2():
    d1 = [14,5]
    d2 = [6,7]
    e1 = [2,10]
    e2 = [8,6]
    c1 = [5,17]
    c2 = [15,5]
    
    
    classes = [d1, d2, e1, e2, c1, c2]

    n = sum([sum(c) for c in classes])

	# Do it the fast way
    ents = []
    for c in classes:
    	ni = sum(c)
    	ent_c = entropy(c)

    	ents.append((ni/n) * ent_c)
    	print(c, ent_c, sep="\t")

    print()
    print(sum(ents))
 
	# Do it the long way, just to make sure they're the same (they are)
    d = combine_entropies(entropy(d1), sum(d1), entropy(d2), sum(d2), sum((sum(d1), sum(d2))))
    e = combine_entropies(entropy(e1), sum(e1), entropy(e2), sum(e2), sum((sum(e1), sum(e2))))
    c = combine_entropies(entropy(c1), sum(c1), entropy(c2), sum(c2), sum((sum(c1), sum(c2))))
    b = combine_entropies(d, sum((sum(d1), sum(d2))), e, sum((sum(e1), sum(e2))), sum((sum(d1), sum(d2), sum(e1), sum(e2))))
    a = combine_entropies(b, sum((sum(d1), sum(d2), sum(e1), sum(e2))), c, sum((sum(c1), sum(c2))), n)
    
    print(f"entropy of a is {a}")
    
    num_error = sum([min(c) for c in classes])  
    error_rate = num_error / n
    print(f"error rate: {num_error} / {n} = {error_rate}")  
    
 
	



task2()