def s_algorithm(examples):
    hypothesis = ['Ø']*len(examples[0][:-1])

    for instances in examples:
        if instances[-1].lower() == 'yes':
            for i in range(len(hypothesis)):
                if hypothesis[i] == 'Ø':
                    hypothesis[i] = instances[i]
                elif hypothesis[i] != instances[i]:
                    hypothesis[i] = '?'
    print(hypothesis)
    return 0
examples = [['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']]
c_hypothesis = s_algorithm(examples)
print(c_hypothesis)