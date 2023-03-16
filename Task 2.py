import numpy as np

R = int(input("Enter the number of rows:"))
C = int(input("Enter the number of columns:"))
shape = R * C
print(f"Enter the entries in a single line (separated by space) with shape {shape}: ")

# User input of entries in a
# single line separated by space
entries = list(map(int, input().split()))

# For making the matrix
matrix = np.array(entries).reshape(R, C)

#Output
output = np.count_nonzero(matrix == 1)
print(f"Number of islands is {output}")