
# Python code to illustrate split() function
inp = input().split()
count = 0
with open("sample_output_cascading.txt", "r") as file:
    lines = file.readlines()
    for i in range(len(lines)):
        line = lines[i].split()
        for j in range(len(inp)):
            if line[j] == inp[j]:
                count +=1 
                continue
            else:
                print("Err")
                break
    if count == len(inp):
        print("Test case passed")
