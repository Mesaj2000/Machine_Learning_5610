from scipy.stats.stats import pearsonr

vectors = {
            "A" : [1,1],
            "B" : [2,1],
            "C" : [3,3],
            "D" : [6,3] }


origin = [0,0]


# Returns the correlation of two vectors with the origin?
def corr(p,q):
    x = [v[0] for v in (origin, p, q)]
    y = [v[1] for v in (origin, p, q)]
    return pearsonr(x,y)[0]


if __name__ == "__main__":
    corrs = dict()
    for pair in "AB AC AD CD BD BC".split():
        x, y = pair[0], pair[1]
        corrs[f"{x},{y}"] = corr(vectors[x], vectors[y])
        
    items = sorted(list(corrs.items()), key=lambda x: x[1], reverse=True)
    
    for key, value in items:
        print(f"Corr({key}) = {value.round(3)}")
