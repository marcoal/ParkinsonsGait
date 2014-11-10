from matplotlib import pyplot

lines = open('sensorPositions.txt').readlines()
lines = [line.strip('\n').split() for line in lines]

for line in lines:
	side, x, y = line
	label = 'b+' if 'L' in side else 'r+'
	pyplot.plot(x, y, label)

pyplot.xlim(-800, 800)
pyplot.ylim(-900, 900)
pyplot.title('Position of foot sensors')
pyplot.show()