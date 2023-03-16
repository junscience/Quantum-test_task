# input number
number = int(input("Enter a number "))

# check if value is correct
assert number > 0, "The input must be positive number!"

# initialize sum value
sum = 0

# for loop
for i in range(number+1):
    sum += i

# output
print(f'Sum value is {sum}')


