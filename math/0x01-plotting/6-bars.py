#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

fig, ax = plt.subplots()

people_labels = ['Farrah', 'Fred', 'Felicia']
fruits_labels = ('apples', 'bananas', 'oranges', 'peaches')
colors = ('red', 'yellow', '#ff8000', '#ffe5b4')
width = 0.5
sub_total = np.zeros(3)

for i, n_fruits in enumerate(fruit):
    ax.bar(
        people_labels,
        n_fruits,
        width,
        bottom=sub_total,
        label=fruits_labels[i],
        color=colors[i]
    )
    sub_total += n_fruits

ax.set_ylabel('Quantity of Fruit')
ax.set_ylim(0, 80)
ax.set_title('Number of Fruit per Person')
ax.legend()

plt.show()
