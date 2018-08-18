def log(string, level=0):
    arrow = ''
    for i in range(1, level + 1):
        arrow += '--'
    if len(arrow):
        arrow += '>'

    print(arrow, ' ', string)
