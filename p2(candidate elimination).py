import csv

def load_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)[1:]  
    return data

def candidate_elimination(data):
    n = len(data[0]) - 1  
    S = ['Ø'] * n         
    G = [['?'] * n]       
    for row in data:
        x, y = row[:-1], row[-1]  

        if y.lower() == 'yes':  
            for i in range(n):
                if S[i] == 'Ø':
                    S[i] = x[i]
                elif S[i] != x[i]:
                    S[i] = '?'

            G = [g for g in G if all(g[i] == '?' or g[i] == S[i] for i in range(n))]

        else:  
            G_new = []
            for g in G:
                for i in range(n):
                    if g[i] == '?':
                        g_copy = g[:]
                        g_copy[i] = S[i]
                        if g_copy not in G_new:
                            G_new.append(g_copy)
            G = G_new

    return S, G

filename = 'enjoysport.csv'  
examples = load_data(filename)

S_final, G_final = candidate_elimination(examples)

print("Final Specific Hypothesis (S):", S_final)
print("Final General Hypotheses (G):")
for g in G_final:
    print(g)
