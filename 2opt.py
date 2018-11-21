#credits : https://stackoverflow.com/questions/25585401/travelling-salesman-in-scipy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
path_distance = lambda r,c: np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))])


# Reverse the order of all elements from element i to element k in array r.
two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))

def two_opt(cities,improvement_threshold):

    route = np.arange(cities.shape[0]) # Make an array of row numbers corresponding to cities.

    improvement_factor = 1 # Initialize the improvement factor.

    best_distance = path_distance(route,cities) # Calculate the distance of the initial path.

    while improvement_factor > improvement_threshold: # If the route is still improving, keep going!

        distance_to_beat = best_distance # Record the distance at the beginning of the loop.

        for swap_first in range(1,len(route)-2): # From each city except the first and last,

            for swap_last in range(swap_first+1,len(route)): # to each of the cities following,

                new_route = two_opt_swap(route,swap_first,swap_last) # try reversing the order of these cities
                new_distance = path_distance(new_route,cities) # and check the total distance with this modification.
                if new_distance < best_distance: # If the path distance is an improvement,
                    print("New distance= ",new_distance)
                    print("new route =",new_route)
                    route = new_route # make this the accepted best route
                    best_distance = new_distance # and update the distance corresponding to this route.

        improvement_factor = 1 - best_distance/distance_to_beat # Calculate how much the route has improved.

    return route # When the route is no longer improving substantially, stop searching and return the route.


# Create a matrix of cities, with each row being a location in 2-space (function works in n-dimensions).
cities = np.random.RandomState(42).rand(30,2)

#Custom instance
C=[[0,3,4,6,8,9,8,10],
   [3,0,5,4,8,6,12,8],
   [4,5,0,2,2,3,5,7],
   [6,4,2,0,3,2,5,4],
   [8,8,2,3,0,2,2,4],
   [9,6,3,2,2,0,3,2],
   [8,12,5,5,2,3,0,2],
   [10,9,7,4,4,2,2,0]]

pca = PCA(n_components=2)
X3d = pca.fit_transform(C)
print(X3d)
# cities=X3d

# Find a good route with 2-opt ("route" gives the order in which to travel to each city by row number.)
route = two_opt(cities,0.001)
print(route)



# Reorder the cities matrix by route order in a new matrix for plotting.
new_cities_order = np.concatenate((np.array([cities[route[i]] for i in range(len(route))]),np.array([cities[0]])))
# Plot the cities.
plt.scatter(cities[:,0],cities[:,1])
# Plot the path.
plt.plot(new_cities_order[:,0],new_cities_order[:,1])
plt.show()

# Print the route as row numbers and the total distance travelled by the path.
print("Route: " + str(route) + "\n\nDistance: " + str(path_distance(route,cities)))
